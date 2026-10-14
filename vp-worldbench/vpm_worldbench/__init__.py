"""VP-WorldBench: multi-view 4D benchmark toolkit.

    from vpm_worldbench import Benchmark, Protocol
    bench = Benchmark("/path/to/VP-WorldBench")
    s = bench.sample("Real-General/C001")
"""
from .constants import KIND_NAMES, SUBSETS
from .data.benchmark import Benchmark
from .data.sample import Cameras, Sample, Tracks
from .protocol import Protocol, get_protocol

__version__ = "1.0.0"
__all__ = ["Benchmark", "Sample", "Cameras", "Tracks", "Protocol", "get_protocol", "SUBSETS", "KIND_NAMES",
           "__version__"]
