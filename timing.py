"""Wall-clock timings of the pipeline stages, written to metrics.json.

One clock (time.perf_counter) and one nested dict of seconds.  Stage names
use dots for nesting: ``with timings.stage("comparisons.oracle"):``.
"""

import os
import time
from contextlib import contextmanager


class Timings:
    def __init__(self):
        self._start = time.perf_counter()
        self._open = {}
        self.data = {}

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def add(self, name: str, seconds: float) -> None:
        """Accumulate `seconds` under the dotted key `name`."""
        *path, leaf = name.split(".")
        node = self.data
        for key in path:
            node = node.setdefault(key, {})
        node[leaf] = node.get(leaf, 0.0) + float(seconds)

    def set(self, name: str, value) -> None:
        *path, leaf = name.split(".")
        node = self.data
        for key in path:
            node = node.setdefault(key, {})
        node[leaf] = value

    def start(self, name: str) -> None:
        """Open stage `name`; for long blocks where `with` would re-indent."""
        self._open[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        self.add(name, time.perf_counter() - self._open.pop(name))

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t)

    def get(self, name: str):
        node = self.data
        for key in name.split("."):
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node

    def finish(self, disjoint: list[str]) -> dict:
        """Close the clock and return the block.

        `disjoint` lists dotted keys whose seconds do not overlap and
        together cover the run; `other` is the untimed rest.  Keys that
        were never timed are skipped.
        """
        total = self.elapsed()
        timed = sum(v for v in (self.get(k) for k in disjoint) if v is not None)
        self.data["other"] = total - timed
        self.data["total"] = total
        return self.data


def environment() -> dict:
    import torch
    try:
        load = list(os.getloadavg())
    except OSError:
        load = None
    return {"torch_threads": torch.get_num_threads(),
            "cpu_count": os.cpu_count(), "loadavg": load}


class SimClock:
    """Per-system accumulators for the inside of `simulate`.

    Wrap each system's draws and step in a zero-argument callable, so the
    draws are charged to that system and their order is unchanged.
    """

    def __init__(self):
        self._start = time.perf_counter()
        self.seconds = {}

    def run(self, name: str, fn):
        """Call the zero-argument `fn`, charging its time to `name`."""
        t = time.perf_counter()
        out = fn()
        self.seconds[name] = (self.seconds.get(name, 0.0)
                              + time.perf_counter() - t)
        return out

    def result(self) -> dict:
        out = dict(self.seconds)
        out["loop"] = time.perf_counter() - self._start
        return out
