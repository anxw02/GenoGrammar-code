import torch.nn as nn
import torch.nn.functional as F

from .common import (
    MatchedEncoderBase,
    MODEL_DIM,
)


class MeanPoolEncoder(
    MatchedEncoderBase
):

    def __init__(self):
        super().__init__()

        self.mean_ff = nn.Sequential(
            nn.Linear(
                MODEL_DIM,
                MODEL_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(
                MODEL_DIM,
            ),
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
        # Strictly permutation-invariant over gene order.
        x = self.mean_ff(
            token
        )

        pooled = x.mean(
            dim=1
        )

        z = self.out_proj(
            pooled
        )

        return F.normalize(
            z,
            dim=-1,
        )
