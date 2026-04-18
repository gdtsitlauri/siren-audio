from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from siren.defenses.detector import DefenseSuite
from siren.streaming import StreamingDefense


@dataclass(slots=True)
class LatencyReport:
    sample_rate: int
    chunk_ms: int
    offline_mean_ms: float
    streaming_mean_ms: float
    meets_10ms_target: bool


def benchmark_latency(
    audio: np.ndarray,
    sample_rate: int = 16_000,
    chunk_ms: int = 20,
    runs: int = 5,
) -> LatencyReport:
    suite = DefenseSuite()
    streaming = StreamingDefense(sample_rate=sample_rate, window_ms=chunk_ms, history_size=4)
    chunk_samples = max(1, int(sample_rate * chunk_ms / 1000.0))

    offline_times = []
    for _ in range(max(1, runs)):
        start = perf_counter()
        suite.analyze(audio, sample_rate)
        offline_times.append((perf_counter() - start) * 1000.0)

    streaming_times = []
    for _ in range(max(1, runs)):
        for idx in range(0, len(audio), chunk_samples):
            chunk = audio[idx : idx + chunk_samples]
            start = perf_counter()
            streaming.process_chunk(chunk)
            streaming_times.append((perf_counter() - start) * 1000.0)

    offline_mean = float(np.mean(offline_times)) if offline_times else 0.0
    streaming_mean = float(np.mean(streaming_times)) if streaming_times else 0.0
    return LatencyReport(
        sample_rate=sample_rate,
        chunk_ms=chunk_ms,
        offline_mean_ms=offline_mean,
        streaming_mean_ms=streaming_mean,
        meets_10ms_target=streaming_mean < 10.0,
    )
