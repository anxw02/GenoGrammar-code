import torch
import torch.nn as nn
import torch.nn.functional as F


FAMILY_DIM = 480
MODEL_DIM = 128
STRAND_DIM = 8
WINDOW_LEN = 64


class MatchedEncoderBase(nn.Module):

    def __init__(self):
        super().__init__()

        # ------------------------------------------------------
        # These modules intentionally match GenoGrammar exactly.
        # Shared parameter count = 195,221.
        # ------------------------------------------------------

        self.family_proj = nn.Sequential(
            nn.Linear(
                FAMILY_DIM,
                MODEL_DIM,
            ),
            nn.LayerNorm(
                MODEL_DIM,
            ),
        )

        self.strand_emb = nn.Embedding(
            2,
            STRAND_DIM,
        )

        self.token_fuse = nn.Sequential(
            nn.Linear(
                MODEL_DIM + STRAND_DIM,
                MODEL_DIM,
            ),
            nn.LayerNorm(
                MODEL_DIM,
            ),
        )

        # Retained because the common R4 evaluator references it.
        # W_SHUFFLE=0 in formal matched-baseline training.
        self.shuffle_head = nn.Sequential(
            nn.Linear(
                MODEL_DIM,
                MODEL_DIM,
            ),
            nn.GELU(),
            nn.Linear(
                MODEL_DIM,
                3,
            ),
        )

        # Token-level auxiliary heads identical across architectures.
        self.adj_head = nn.Sequential(
            nn.Linear(
                4 * MODEL_DIM,
                MODEL_DIM,
            ),
            nn.GELU(),
            nn.Linear(
                MODEL_DIM,
                2,
            ),
        )

        self.query_proj = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
            bias=False,
        )

        self.key_proj = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
            bias=False,
        )


    @staticmethod
    def pair_feature(
        a,
        b,
    ):
        return torch.cat(
            [
                a,
                b,
                b - a,
                a * b,
            ],
            dim=-1,
        )


    def prepare_tokens(
        self,
        family_embedding,
        strand,
    ):
        x = self.family_proj(
            family_embedding
        )

        s = self.strand_emb(
            strand
        )

        x = self.token_fuse(
            torch.cat(
                [x, s],
                dim=-1,
            )
        )

        return x


    def encode_tokens(
        self,
        token,
    ):
        raise NotImplementedError


    def forward(
        self,
        family_embedding,
        strand,
    ):
        token = self.prepare_tokens(
            family_embedding,
            strand,
        )

        return self.encode_tokens(
            token
        )
