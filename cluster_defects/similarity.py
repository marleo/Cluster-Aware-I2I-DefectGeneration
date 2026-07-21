from __future__ import annotations

import torch
import torch.nn.functional as F


def top_k_mean_similarity(
    queries: torch.Tensor,
    references: torch.Tensor,
    k: int,
) -> torch.Tensor:
    queries = F.normalize(queries.float(), dim=1)
    references = F.normalize(references.float(), dim=1)
    similarities = (queries @ references.T).clamp(-1.0, 1.0)
    effective_k = min(k, references.shape[0])
    return similarities.topk(effective_k, dim=1).values.mean(dim=1)


def maximum_similarity(
    queries: torch.Tensor,
    references: torch.Tensor,
) -> torch.Tensor:
    queries = F.normalize(queries.float(), dim=1)
    references = F.normalize(references.float(), dim=1)
    return (queries @ references.T).clamp(-1.0, 1.0).max(dim=1).values


def leave_one_out_scores(
    embeddings: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(embeddings) < 2:
        raise ValueError("At least two embeddings are required")

    normalized = F.normalize(embeddings.float(), dim=1)
    similarities = (normalized @ normalized.T).clamp(-1.0, 1.0)
    similarities.fill_diagonal_(-torch.inf)
    effective_k = min(k, len(embeddings) - 1)
    top_k = similarities.topk(effective_k, dim=1).values.mean(dim=1)
    maximum = similarities.max(dim=1).values
    return top_k, maximum


def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: list[float] | None = None,
) -> float:
    x = F.normalize(x.float(), dim=1)
    y = F.normalize(y.float(), dim=1)
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    if bandwidths is None:
        combined = torch.cat([x, y], dim=0)
        pairwise = torch.cdist(combined, combined).pow(2)
        positive = pairwise[pairwise > 0]
        median = float(torch.median(positive)) if len(positive) else 1.0
        bandwidths = [median / 4, median / 2, median, median * 2, median * 4]

    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    values = []
    for bandwidth in bandwidths:
        gamma = 1.0 / (2.0 * max(float(bandwidth), 1e-8))
        values.append(
            torch.exp(-gamma * xx).mean()
            + torch.exp(-gamma * yy).mean()
            - 2.0 * torch.exp(-gamma * xy).mean()
        )
    return float(torch.stack(values).mean())
