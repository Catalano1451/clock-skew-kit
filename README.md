# clock-skew-kit

Estimates the clock skew of a remote server by analyzing paired timestamp samples with a robust median-based estimator.

```python
from clock_skew_kit import SkewEstimator, SkewResult

estimator = SkewEstimator()
estimator.add_sample(105.0, 100.0)
estimator.add_sample(106.0, 101.0)
estimator.add_sample(107.0, 102.0)

result: SkewResult = estimator.estimate()
print(result.skew_seconds, result.median_absolute_deviation)
```

## Why this exists

When you receive a timestamp from a remote server, the value it reports may disagree with your local clock. That disagreement comes from two sources: a constant offset (the remote clock was simply set to a different epoch instant) and drift (the remote clock runs at a slightly different rate). This library estimates only the offset. It uses the median of `(remote - local)` differences across samples because network jitter produces outliers, and the median tolerates up to 50% contamination without breaking. We made this choice deliberately over linear regression, which would model drift but requires careful windowing and is far more sensitive to outliers.

## The awkward edge

A single sample produces a skew estimate with a median absolute deviation of zero, which looks like high confidence but is not. Callers should treat `sample_count` as a gate: a handful of samples is not enough to trust the estimate. There is no built-in minimum-sample enforcement because the right threshold depends on your jitter environment.
