import torch
import torch.nn as nn
from torch.nn import functional as F


class CosineContrastiveLoss(nn.Module):
    """
    Assume the input is of shape (batch_size, embedding_dim), i.e., single vector per sample.
    Used cosine similiarity as the similarity measure.
    """

    def __init__(self, margin: float = 0.5, num_negatives: int = 3):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin
        self.num_negatives = num_negatives

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        # anchor [B, D]
        # positive [B, D]
        # negative [B*num_negative, D]
        anchor = anchor[:, 0, :]
        positive = positive[:, 0, :]
        negative = negative[:, 0, :]

        # Normalize the vectors
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)

        # Compute the cosine similarity
        sim_pos = torch.exp(
            F.cosine_similarity(anchor, positive, dim=-1) / self.margin
        )  # [B, 1]
        sim_neg = (
            F.cosine_similarity(
                anchor.unsqueeze(0),
                negative.view(-1, self.num_negatives, negative.size(-1)).permute(
                    1, 0, 2
                ),
                dim=-1,
            )
            / self.margin
        )  # [num_negatives, B]
        sim_neg = torch.exp(sim_neg).sum(dim=0, keepdim=True).permute(1, 0)  # [B, 1]

        # Compute the loss
        loss = torch.log(1 + sim_neg / sim_pos).mean()

        return loss


class AvgSimContrastiveLoss(nn.Module):
    """
    Assumen the input is of shape (batch_size, num_tokens, embedding_dim), i.e., multiple vectors per sample.
    Used cosine similiarity as the similarity measure.
    Average the similarity scores of all the tokens for a particular sample.
    """

    def __init__(self, margin: float = 0.5, num_negatives: int = 3):
        super(AvgSimContrastiveLoss, self).__init__()
        self.margin = margin
        self.num_negatives = num_negatives

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:

        B, T, D = anchor.shape

        # Normalize the vectors
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)

        anchor = anchor.unsqueeze(1)
        positive = positive.unsqueeze(1)
        negative = negative.view(B, -1, T, D)
        positive_score = torch.matmul(anchor, positive.permute(0, 1, 3, 2))
        positive_score = torch.mean(torch.max(positive_score, dim=-2)[0], dim=-1)

        negative_score = torch.matmul(anchor, negative.permute(0, 1, 3, 2))
        negative_score = torch.mean(negative_score, dim=(-1, -2))

        sim_neg = (
            torch.exp(negative_score / self.margin)
            .sum(dim=1, keepdim=True)
            .permute(1, 0)
        )
        sim_pos = torch.exp(positive_score / self.margin)

        loss = torch.log(1 + sim_neg / sim_pos).mean()
        return loss


class MaxSimContrastiveLoss(nn.Module):
    """
    Assumen the input is of shape (batch_size, num_tokens, embedding_dim), i.e., multiple vectors per sample.
    Used cosine similiarity as the similarity measure.
    Maximum among similarity scores of all the tokens for a particular sample.
    """

    def __init__(self, margin: float = 0.5, num_negatives: int = 3):
        super(MaxSimContrastiveLoss, self).__init__()
        self.margin = margin
        self.num_negatives = num_negatives

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        B, T, D = anchor.shape

        # Normalize the vectors
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)

        anchor = anchor.unsqueeze(1)
        positive = positive.unsqueeze(1)
        negative = negative.view(B, -1, T, D)
        positive_score = torch.matmul(anchor, positive.permute(0, 1, 3, 2))
        positive_score = torch.mean(torch.max(positive_score, dim=-2)[0], dim=-1)

        negative_score = torch.matmul(anchor, negative.permute(0, 1, 3, 2))
        negative_score = torch.mean(torch.max(negative_score, dim=-2)[0], dim=-1)

        sim_neg = (
            torch.exp(negative_score / self.margin)
            .sum(dim=1, keepdim=True)
            .permute(1, 0)
        )
        sim_pos = torch.exp(positive_score / self.margin)

        loss = torch.log(1 + sim_neg / sim_pos).mean()
        return loss


class TripletLoss(nn.Module):
    """
    Applicable for both single and multiple vectors per sample.
    L2 Norm based similarity measure.
    Difference between the positive and negative samples should be greater than the margin.
    """

    def __init__(self, margin: float = 1.0, num_negatives: int = 3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.num_negatives = num_negatives

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        # anchor [B, D] or [B, num_tokens, D]
        # positive [B, D] or [B, num_tokens, D]
        # negative [B*num_negative, D] or [B*num_negative, num_tokens, D]
        # anchor = anchor[:, 0, :]
        # positive = positive[:, 0, :]
        # negative = negative[:, 0, :]

        # Normalize the vectors
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)
        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)
            positive = positive.unsqueeze(1)
            negative = negative.unsqueeze(1)
        neg_temp = negative.view(
            -1, self.num_negatives, negative.size(-2), negative.size(-1)
        ).permute(
            1, 0, 2, 3
        )  # [num_negatives, B, num_tokens, D]
        pos_dist = torch.norm(anchor - positive, p=2, dim=-1)
        neg_dist = torch.norm(anchor.unsqueeze(0) - neg_temp, p=2, dim=-1).mean(dim=0)
        loss = (pos_dist.mean(dim=-1) - neg_dist.mean(dim=-1) + self.margin).mean()
        return loss
