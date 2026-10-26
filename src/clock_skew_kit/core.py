"""Core clock-skew estimation logic.

This module estimates clock skew from paired (remote, local) timestamps.
The model is deliberately simple: we assume a linear relationship

    remote_time = local_time + skew + noise

and estimate `skew` as the median of per-sample differences
(remote - local). We chose the median (not the mean) because timestamp
samples can contain outliers from network jitter, and the median is
robust to up to 50% of those.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


def _median(values: List[float]) -> float:
    """Return the median of a non-empty list of floats.

    We implement this here rather than relying on `statistics.median`
    so the behaviour (interpolating the two middle values for an
    even-length list) is explicit and documented.

    Raises:
        ValueError: if `values` is empty.
    """
    if not values:
        raise ValueError("median of empty list is undefined")
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    # Even length: average the two middle elements.
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass(frozen=True)
class SkewResult:
    """Result of a clock-skew estimate.

    Attributes:
        skew_seconds: Estimated skew in seconds. A positive value means
            the remote clock reads later than the local clock, on average.
        sample_count: Number of timestamp pairs used.
        median_absolute_deviation: MAD of (remote - local) differences,
            a robust measure of spread that helps judge confidence.
    """

    skew_seconds: float
    sample_count: int
    median_absolute_deviation: float


class SkewEstimator:
    """Estimate remote clock skew from paired timestamps.

    Call :meth:`add_sample` for each pair of (remote, local) timestamps,
    then call :meth:`estimate` to obtain a :class:`SkewResult`.

    Interpretation decision (stated plainly):
        "Skew" is defined here as `remote - local`. A positive result
        means the remote clock is ahead of the local clock. We do NOT
        model clock *drift* (rate difference over time); we estimate a
        constant offset only.
    """

    def __init__(self) -> None:
        self._samples: List[Tuple[float, float]] = []

    def add_sample(self, remote_timestamp: float, local_timestamp: float) -> None:
        """Record a single timestamp pair.

        Args:
            remote_timestamp: A timestamp reported by the remote server,
                expressed as a Unix epoch float (seconds since 1970-01-01 UTC).
            local_timestamp: The local-clock reading taken as close as
                possible to the moment the remote timestamp was observed.

        Raises:
            ValueError: If either timestamp is not finite.
        """
        self._validate_timestamp(remote_timestamp, "remote")
        self._validate_timestamp(local_timestamp, "local")
        self._samples.append((float(remote_timestamp), float(local_timestamp)))

    @staticmethod
    def _validate_timestamp(value: float, label: str) -> None:
        """Ensure a timestamp is a real, finite number.

        Rejects NaN and infinities. Negative values are permitted because
        some systems use epoch-relative offsets that can be negative.
        """
        if isinstance(value, bool):
            raise ValueError(f"{label} timestamp must be a number, got bool")
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} timestamp must be a number")
        if v != v:  # NaN check
            raise ValueError(f"{label} timestamp must be finite, got NaN")
        if v == float("inf") or v == float("-inf"):
            raise ValueError(f"{label} timestamp must be finite, got {v}")

    def estimate(self) -> SkewResult:
        """Estimate skew from the collected samples.

        Returns:
            A :class:`SkewResult` with the median-based skew estimate and
            the median absolute deviation as a confidence indicator.

        Raises:
            ValueError: If no samples have been added.
        """
        if not self._samples:
            raise ValueError("no samples available; call add_sample first")

        differences: List[float] = [
            remote - local for remote, local in self._samples
        ]
        skew = _median(differences)

        deviations = [abs(d - skew) for d in differences]
        mad = _median(deviations)

        return SkewResult(
            skew_seconds=skew,
            sample_count=len(self._samples),
            median_absolute_deviation=mad,
        )

    def reset(self) -> None:
        """Discard all samples so the estimator can be reused."""
        self._samples.clear()
