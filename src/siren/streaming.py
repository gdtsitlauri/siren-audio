from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from siren.defenses.detector import DefenseReport, DefenseSuite


@dataclass(slots=True)
class StreamingDecision:
    blocked: bool
    average_risk: float
    latest: DefenseReport


class StreamingDefense:
    """Sliding-window defense wrapper for low-latency analysis."""

    def __init__(self, sample_rate: int, window_ms: int = 250, history_size: int = 5) -> None:
        self.sample_rate = sample_rate
        self.window_samples = max(32, int(sample_rate * window_ms / 1000.0))
        self.history: deque[float] = deque(maxlen=max(1, history_size))
        self.suite = DefenseSuite()

    def process_chunk(self, chunk: np.ndarray) -> StreamingDecision:
        chunk = np.asarray(chunk, dtype=np.float32)
        if len(chunk) < self.window_samples:
            chunk = np.pad(chunk, (0, self.window_samples - len(chunk)))
        else:
            chunk = chunk[: self.window_samples]
        report = self.suite.analyze(chunk, self.sample_rate)
        self.history.append(report.risk_score)
        average = float(np.mean(list(self.history)))
        return StreamingDecision(blocked=average >= 0.5, average_risk=average, latest=report)
