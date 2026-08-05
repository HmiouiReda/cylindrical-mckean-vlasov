"""Benchmark registry.

Adding a benchmark = one file implementing `benchmarks.base.Benchmark`
plus one entry in the two dictionaries below (and, optionally, a JSON file
in configs/).
"""

from .base import Benchmark
from .ou import OrnsteinUhlenbeck, OUParams
from .double_well import DoubleWell, DoubleWellParams
from .multiplicative import MultiplicativeOU, MultiplicativeParams
from .ou2d import OU2d, OU2dParams
from .ou_non_gaussian import UniformInitialOU,UniformOUParams,GaussianMixtureInitialOU,GaussianMixtureOUParams

REGISTRY = {
    "ou": OrnsteinUhlenbeck,
    "double_well": DoubleWell,
    "multiplicative": MultiplicativeOU,
    "ou2d": OU2d,
    "ou_uniform": UniformInitialOU,
    "ou_gmm": GaussianMixtureInitialOU,
}

PARAMS = {
    "ou": OUParams,
    "double_well": DoubleWellParams,
    "multiplicative": MultiplicativeParams,
    "ou2d": OU2dParams,
    "ou_uniform": UniformOUParams,
    "ou_gmm": GaussianMixtureOUParams,
}


def get_benchmark(name: str, **params) -> Benchmark:
    if name not in REGISTRY:
        raise KeyError(f"unknown benchmark '{name}' "
                       f"(available: {sorted(REGISTRY)})")
    return REGISTRY[name](**params)
