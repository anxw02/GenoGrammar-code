import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    MatchedEncoderBase,
    MODEL_DIM,
)


class CNN1DEncoder(
    MatchedEncoderBase
):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            MODEL_DIM,
            MODEL_DIM,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv1d(
            MODEL_DIM,
            MODEL_DIM,
            kernel_size=5,
            padding=2,
        )

        self.conv3 = nn.Conv1d(
            MODEL_DIM,
            MODEL_DIM,
            kernel_size=3,
            padding=1,
        )

        self.conv4 = nn.Conv1d(
            MODEL_DIM,
            MODEL_DIM,
            kernel_size=5,
            padding=2,
        )

        self.core_norm = nn.LayerNorm(
            MODEL_DIM
        )

        # Brings total capacity extremely close to GenoGrammar.
        self.core_ff = nn.Sequential(
            nn.Linear(
                MODEL_DIM,
                MODEL_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(
                MODEL_DIM,
            ),
            nn.Linear(
                MODEL_DIM,
                MODEL_DIM,
            ),
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


    def encode_tokens(
        self,
        token,
    ):
        x = token.transpose(
            1,
            2,
        )

        x = F.gelu(
            self.conv1(x)
        )

        x = F.gelu(
            self.conv2(x)
        )

        x = F.gelu(
            self.conv3(x)
        )

        x = F.gelu(
            self.conv4(x)
        )

        x = x.transpose(
            1,
            2,
        )

        x = self.core_norm(
            x
        )

        x = (
            x
            +
            self.core_ff(x)
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
