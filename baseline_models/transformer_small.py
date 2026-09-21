import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    MatchedEncoderBase,
    MODEL_DIM,
    WINDOW_LEN,
)


class SmallTransformerEncoder(
    MatchedEncoderBase
):

    def __init__(self):
        super().__init__()

        self.position = nn.Parameter(
            torch.zeros(
                1,
                WINDOW_LEN,
                MODEL_DIM,
            )
        )

        layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=4,
            dim_feedforward=256,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
        )

        self.core_norm = nn.LayerNorm(
            MODEL_DIM
        )

        self.pool_gate = nn.Linear(
            MODEL_DIM,
            1,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(
                MODEL_DIM,
                MODEL_DIM,
            ),
            nn.LayerNorm(
                MODEL_DIM,
            ),
        )

        self.post_ff = nn.Sequential(
            nn.Linear(
                MODEL_DIM,
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
        L = token.shape[1]

        if L > self.position.shape[1]:
            raise ValueError(
                f"Sequence length {L} exceeds "
                f"maximum {self.position.shape[1]}"
            )

        x = (
            token
            +
            self.position[
                :,
                :L,
            ]
        )

        x = self.encoder(
            x
        )

        x = self.core_norm(
            x
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

        z = (
            z
            +
            self.post_ff(z)
        )

        return F.normalize(
            z,
            dim=-1,
        )
