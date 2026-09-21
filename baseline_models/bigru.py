import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    MatchedEncoderBase,
    MODEL_DIM,
)


class BiGRUEncoder(
    MatchedEncoderBase
):

    HIDDEN = 94

    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=MODEL_DIM,
            hidden_size=self.HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=0.10,
            bidirectional=True,
        )

        width = (
            2
            *
            self.HIDDEN
        )

        self.pool_gate = nn.Linear(
            width,
            1,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(
                width,
                MODEL_DIM,
            ),
            nn.LayerNorm(
                MODEL_DIM,
            ),
        )


    def encode_tokens(
        self,
        token,
    ):
        x, _ = self.gru(
            token
        )

        weight = torch.softmax(
            self.pool_gate(x)
            .squeeze(-1),
            dim=1,
        )

        pooled = (
            x
            *
            weight.unsqueeze(-1)
        ).sum(
            dim=1
        )

        z = self.out_proj(
            pooled
        )

        return F.normalize(
            z,
            dim=-1,
        )
