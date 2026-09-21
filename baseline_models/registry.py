from .mean_pool import MeanPoolEncoder
from .cnn1d import CNN1DEncoder
from .bigru import BiGRUEncoder
from .transformer_small import SmallTransformerEncoder


REGISTRY = {
    "mean_pool":
        MeanPoolEncoder,

    "cnn1d":
        CNN1DEncoder,

    "bigru":
        BiGRUEncoder,

    "transformer_small":
        SmallTransformerEncoder,
}


def build_model(
    name,
):
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown baseline: {name}. "
            f"Available={list(REGISTRY)}"
        )

    return REGISTRY[name]()
