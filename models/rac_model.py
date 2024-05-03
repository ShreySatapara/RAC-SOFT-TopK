import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from .soft_topk import SoftTopK
from .scores import CosineScores, DotProductScores, L2Scores, MaxSimScores, AvgSimScores
from .bert_seq_cls import BertForSequenceClassification, BertModel
import os
import json

SCORE_DICT = {
    "cosine": CosineScores,
    "dot_product": DotProductScores,
    "l2": L2Scores,
    "maxsim": MaxSimScores,
    "avgsim": AvgSimScores,
}


# Density based cross-entropy loss
class DensityLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DensityLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, A, A_true):
        if A_true is None:
            return torch.tensor(0).to(A.device)
        # print(A.shape, A_true.shape)
        B, N, M = A_true.shape
        A = A[:, :, :M]  # Truncate A to match A_true
        A = A + self.epsilon
        A_true = A_true + self.epsilon
        loss = (A_true * torch.log(A_true / A)).sum() / (B * N)
        return loss


class RACModel(nn.Module):
    def __init__(
        self,
        config: Namespace,
    ):
        super(RACModel, self).__init__()
        # RETRIEVER
        retriever_path = (
            config.retriever_path
            if config.retriever_path != "None"
            else config.bert_model_name_retriever
        )
        self.retriever = BertModel.from_pretrained(retriever_path)
        # CLASSIFIER
        classifier_path = (
            config.classifier_path
            if config.classifier_path != "None"
            else config.bert_model_name_classifier
        )
        self.classifier = BertForSequenceClassification.from_pretrained(
            classifier_path, num_labels=config.num_labels
        )
        self.classifier_embeddings = self.classifier.bert.embeddings

        # SOFT TOPK
        self.soft_topk = SoftTopK(config.retriever_count, epsilon=config.soft_k_epsilon)
        self.density_loss = DensityLoss(config.epsilon)
        self.loss_fn = nn.CrossEntropyLoss()

        # SCORE FUNCTION
        self.score_fn = SCORE_DICT[config.score_fn]()

        # Retriever Count is the number of documents to retrieve
        self.retriever_count = config.retriever_count
        self.config = config
        if config.freeze_retriever:
            for param in self.retriever.parameters():
                param.requires_grad = False
        if config.freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = False

    # Only batch size 1 is supported
    def retrieve_scores(self, query, docs):
        B = query["input_ids"].shape[0]
        # print("Query", query["input_ids"], "Docs", docs["input_ids"])
        query_repr = self.retriever(**query)["last_hidden_state"]
        docs_repr = self.retriever(**docs)["last_hidden_state"]
        # print("Query", torch.isnan(query_repr), "Docs", torch.isnan(docs_repr))
        _, T, D = docs_repr.shape
        docs_repr = docs_repr.reshape(B, -1, T, D)
        # NOTE: Which token to consider
        query_repr = query_repr
        docs_repr = docs_repr
        scores = self.score_fn(query_repr, docs_repr)
        # print("Scores", scores)
        return scores

    def classify(self, query_docs, soft_A):
        B, T = query_docs["input_ids"].shape
        embeddings = self.classifier_embeddings(query_docs["input_ids"])
        embeddings = embeddings.unsqueeze(0)
        embeddings = soft_A.permute(0, 2, 1).matmul(embeddings.permute(0, 2, 1, 3))
        embeddings = embeddings.permute(0, 2, 1, 3)
        embeddings = embeddings.squeeze(0)
        outputs = self.classifier(
            input_ids=None,
            inputs_embeds=embeddings,
        )
        return outputs

    def forward(self, inputs, return_outputs=False):
        query = inputs.pop("query")
        docs = inputs.pop("docs")
        query_docs = inputs.pop("query_docs")
        true_soft_A = inputs.pop("soft_A")
        labels = inputs.pop("labels")

        scores = self.retrieve_scores(query, docs)
        assert scores.shape[0] == 1
        soft_A, _ = self.soft_topk(scores)
        outputs = self.classify(query_docs, soft_A)
        soft_A_loss = self.density_loss(soft_A, true_soft_A)  # Loss1: Density Loss
        soft_scores = soft_A.permute(0, 2, 1).matmul(scores.unsqueeze(-1))
        outputs = soft_scores.squeeze(0).t() @ outputs.logits
        mean_scores = soft_scores.mean(dim=1)
        outputs = outputs / mean_scores
        outputs = F.softmax(outputs, dim=-1)
        hard_loss = self.loss_fn(outputs, labels)  # Loss2: Cross Entropy Loss
        loss = soft_A_loss + hard_loss  # Total Loss
        # print("Soft A Loss", soft_A_loss.item(), "Hard Loss", hard_loss.item())
        if return_outputs:
            return loss, outputs
        return loss

    def to_device(self, data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {key: self.to_device(value, device) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self.to_device(item, device) for item in data]
        else:
            return data

    def save_model(self, path):
        self.retriever.save_pretrained(os.path.join(path, "retriever"))
        self.classifier.save_pretrained(os.path.join(path, "classifier"))
        json.dump(vars(self.config), open(os.path.join(path, "model_config.json"), "w"))
