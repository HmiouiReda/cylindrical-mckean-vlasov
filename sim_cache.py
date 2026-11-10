"""Cache of one run's simulation, so runs that differ only in model settings
reuse it.

Stores the simulated scenarios, the evaluation points of the three splits and
the random-generator states right after each was drawn. Restoring those states
makes a cached run take the same split and the same draws as an uncached one.
The key holds everything the simulation depends on and nothing else: not the
functional, not the network settings.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path


def cache_key(benchmark: str, bench_params: dict, M: int, N: int, seed: int,
              k_x: int, k_x_test: int, n_eval_particles: int) -> str:
    blob = json.dumps(dict(benchmark=benchmark, params=bench_params, M=M, N=N,
                           seed=seed, k_x=k_x, k_x_test=k_x_test,
                           n_eval_particles=n_eval_particles),
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class SimCache:
    def __init__(self, directory: str | Path, key: str, label: str = ""):
        self.dir = Path(directory)
        self.path = self.dir / f"{label}{'_' if label else ''}{key}.pkl"

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        with open(self.path, "rb") as f:
            return pickle.load(f)

    def save(self, payload: dict) -> None:
        # temp file then rename, so a parallel reader never sees half a file
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)
