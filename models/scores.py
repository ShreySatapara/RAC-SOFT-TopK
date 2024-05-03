import torch
import torch.nn as nn
import torch.nn.functional as F


class Scores(nn.Module):
    def __init__(self):
        super(Scores, self).__init__()

    def forward(self, query, docs):
        raise NotImplementedError


class CosineScores(Scores):
    def __init__(self):
        super(CosineScores, self).__init__()

    def forward(self, query_repr, docs_repr):
        # print(query_repr.shape, docs_repr.shape)
        query_repr = query_repr[:, 0, :]
        docs_repr = docs_repr[:, :, 0, :]
        B, N, D = docs_repr.shape
        assert query_repr.shape == (B, D)
        query_repr = query_repr.unsqueeze(1)
        query_repr = F.normalize(query_repr, p=2, dim=-1)
        docs_repr = F.normalize(docs_repr, p=2, dim=-1)
        scores = torch.matmul(query_repr, docs_repr.permute(0, 2, 1)).squeeze(-2)
        assert scores.shape == (B, N)
        return scores


class DotProductScores(Scores):
    def __init__(self):
        super(DotProductScores, self).__init__()

    def forward(self, query_repr, docs_repr):
        query_repr = query_repr[:, 0, :]
        docs_repr = docs_repr[:, :, 0, :]
        B, N, D = docs_repr.shape
        assert query_repr.shape == (B, D)
        query_repr = query_repr.unsqueeze(1)
        scores = torch.matmul(query_repr, docs_repr.permute(0, 2, 1)).squeeze(-2)
        assert scores.shape == (B, N)
        return scores


class L2Scores(Scores):
    def __init__(self):
        super(L2Scores, self).__init__()

    def forward(self, query_repr, docs_repr):
        query_repr = query_repr[:, 0, :]
        docs_repr = docs_repr[:, :, 0, :]
        B, N, D = docs_repr.shape
        assert query_repr.shape == (B, D)
        query_repr = query_repr.unsqueeze(1)
        scores = -torch.norm(query_repr - docs_repr, p=2, dim=-1)
        assert scores.shape == (B, N)
        return scores


class MaxSimScores(Scores):
    def __init__(self):
        super(MaxSimScores, self).__init__()

    def forward(self, query_repr, docs_repr):
        B, N, T, D = docs_repr.shape
        assert query_repr.shape == (B, T, D)

        query_repr = query_repr.unsqueeze(1)

        query_repr = F.normalize(query_repr, p=2, dim=-1)
        docs_repr = F.normalize(docs_repr, p=2, dim=-1)

        score_matrix = torch.matmul(query_repr, docs_repr.permute(0, 1, 3, 2))
        scores, _ = torch.max(score_matrix, dim=-2)
        scores = torch.mean(scores, dim=-1)
        assert scores.shape == (B, N)
        return scores


class AvgSimScores(Scores):
    def __init__(self):
        super(AvgSimScores, self).__init__()

    def forward(self, query_repr, docs_repr):
        B, N, T, D = docs_repr.shape
        assert query_repr.shape == (B, T, D)

        query_repr = query_repr.unsqueeze(1)

        query_repr = F.normalize(query_repr, p=2, dim=-1)
        docs_repr = F.normalize(docs_repr, p=2, dim=-1)

        score_matrix = torch.matmul(query_repr, docs_repr.permute(0, 1, 3, 2))
        scores = torch.mean(score_matrix, dim=(-1, -2))
        assert scores.shape == (B, N)
        return scores
