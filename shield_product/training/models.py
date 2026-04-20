from typing import Sequence, Tuple

import torch
from torch import nn


def _normalize_hidden_dims(hidden_dims: Sequence[int]) -> Tuple[int, int]:
    if len(hidden_dims) < 2:
        return 128, 64

    first = int(hidden_dims[0])
    second = int(hidden_dims[1])
    if first <= 0 or second <= 0:
        return 128, 64
    return first, second


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        first_hidden, second_hidden = _normalize_hidden_dims(hidden_dims)
        dropout = max(0.0, min(float(dropout), 0.8))
        self.net = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
