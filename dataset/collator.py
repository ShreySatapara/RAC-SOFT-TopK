import torch
from transformers import DefaultDataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Dict, List, Any
from .utils import tokenize_sample
import numpy as np


class ContrastiveDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def list_to_tensor(
        self, example: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        stack_example = {key: None for key in example[0].keys()}
        for key in stack_example.keys():
            stack_example[key] = torch.stack([x[key].squeeze(0) for x in example])
        return stack_example

    def collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        collated_batch = {"rumor": [], "evidence": [], "negatives": []}

        for sample in batch:
            # 1. Tokenize the sample
            sample = tokenize_sample(
                sample=sample,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            collated_batch["rumor"].append(sample["rumor"])
            collated_batch["evidence"].append(sample["evidence"])
            collated_batch["negatives"].extend(sample["negatives"])

        # 2. Convert the list of tensors to a single tensor
        collated_batch["rumor"] = self.list_to_tensor(collated_batch["rumor"])
        collated_batch["evidence"] = self.list_to_tensor(collated_batch["evidence"])
        collated_batch["negatives"] = self.list_to_tensor(collated_batch["negatives"])

        return collated_batch

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors="pt"
    ) -> Dict[str, Any]:
        if return_tensors == "pt":
            return self.collate_batch(features)
        else:
            raise ValueError(
                "return_tensors should be 'pt' for PyTorch tensors. Others are not implemented."
            )


class ClassifierDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        text, labels = [], []
        for sample in batch:
            text.append(sample["rumor"] + self.tokenizer.sep_token + sample["evidence"])
            labels.append(sample["label"])

        # Tokenize the text
        tokenized_batch = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized_batch["labels"] = torch.tensor(labels)
        return dict(tokenized_batch)

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors=None
    ) -> Dict[str, Any]:
        return self.collate_batch(features)


class JointDataCollator(DefaultDataCollator):
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 256,
    ):
        if isinstance(tokenizer, tuple):
            self.tokenizer_classifier = tokenizer[0]
            self.tokenizer_retriever = tokenizer[1]
        else:
            self.tokenizer_classifier = tokenizer
            self.tokenizer_retriever = tokenizer
        self.max_length = max_length

    def collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        query, docs, query_docs, labels, soft_A = [], [], [], [], []

        for sample in batch:
            query.append(sample["query"])
            docs.extend(sample["docs"])
            for doc in sample["docs"]:
                query_docs.append(
                    sample["query"] + self.tokenizer_classifier.sep_token + doc
                )
            labels.append(sample["label"])
            soft_A.append(sample["soft_A"])

        # Tokenize the text
        collated_batch = {}
        collated_batch["query"] = dict(
            self.tokenizer_retriever(
                query,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        )
        collated_batch["docs"] = dict(
            self.tokenizer_retriever(
                docs,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        )
        collated_batch["query_docs"] = dict(
            self.tokenizer_classifier(
                query_docs,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        )
        collated_batch["labels"] = torch.tensor(labels)
        collated_batch["soft_A"] = (
            None if soft_A[0] is None else torch.tensor(np.array(soft_A))
        )

        return collated_batch

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors=None
    ) -> Dict[str, Any]:
        return self.collate_batch(features)
