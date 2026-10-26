import math
import unittest

from clock_skew_kit import SkewEstimator, SkewResult


class TestSkewEstimator(unittest.TestCase):

    def test_imports_and_exports(self):
        self.assertTrue(hasattr(SkewEstimator, "add_sample"))
        self.assertTrue(hasattr(SkewEstimator, "estimate"))
        self.assertTrue(hasattr(SkewEstimator, "reset"))
        result = SkewResult(skew_seconds=0.0, sample_count=0, median_absolute_deviation=0.0)
        self.assertTrue(hasattr(result, "skew_seconds"))
        self.assertTrue(hasattr(result, "sample_count"))
        self.assertTrue(hasattr(result, "median_absolute_deviation"))

    def test_positive_skew(self):
        # Remote clock is 5 seconds ahead of local at every sample.
        estimator = SkewEstimator()
        estimator.add_sample(remote_timestamp=105.0, local_timestamp=100.0)
        estimator.add_sample(remote_timestamp=106.0, local_timestamp=101.0)
        estimator.add_sample(remote_timestamp=107.0, local_timestamp=102.0)
        result = estimator.estimate()
        self.assertEqual(result.skew_seconds, 5.0)
        self.assertEqual(result.sample_count, 3)
        # All differences identical, so MAD is 0.
        self.assertEqual(result.median_absolute_deviation, 0.0)

    def test_negative_skew(self):
        # Remote clock is 3 seconds behind local.
        estimator = SkewEstimator()
        estimator.add_sample(97.0, 100.0)
        estimator.add_sample(98.0, 101.0)
        result = estimator.estimate()
        self.assertEqual(result.skew_seconds, -3.0)

    def test_robust_to_outlier(self):
        # 4 clean samples at +2s, one outlier at +100s.
        # Median of [2,2,2,2,100] is 2 — robust.
        estimator = SkewEstimator()
        for i in range(4):
            estimator.add_sample(102.0 + i, 100.0 + i)
        estimator.add_sample(200.0, 100.0)
        result = estimator.estimate()
        self.assertEqual(result.skew_seconds, 2.0)

    def test_even_sample_count_interpolation(self):
        # Two samples with differences 1.0 and 3.0.
        # Median of even-length list => (1+3)/2 = 2.0
        estimator = SkewEstimator()
        estimator.add_sample(101.0, 100.0)  # diff 1
        estimator.add_sample(103.0, 100.0)  # diff 3
        result = estimator.estimate()
        self.assertEqual(result.skew_seconds, 2.0)

    def test_odd_sample_count_exact_median(self):
        # Differences: 1, 3, 5 -> median 3
        estimator = SkewEstimator()
        estimator.add_sample(101.0, 100.0)
        estimator.add_sample(103.0, 100.0)
        estimator.add_sample(105.0, 100.0)
        result = estimator.estimate()
        self.assertEqual(result.skew_seconds, 3.0)

    def test_mad_calculation(self):
        # Differences: 1, 2, 10. Median = 2. Deviations: 1, 0, 8. MAD = 1.
        estimator = SkewEstimator()
        estimator.add_sample(101.0, 100.0)
        estimator.add_sample(102.0, 100.0)
        estimator.add_sample(110.0, 100.0)
        result = estimator.estimate()
        self.assertEqual(result.skew_seconds, 2.0)
        self.assertEqual(result.median_absolute_deviation, 1.0)

    def test_estimate_with_no_samples_raises(self):
        estimator = SkewEstimator()
        with self.assertRaises(ValueError):
            estimator.estimate()

    def test_add_sample_nan_remote_raises(self):
        estimator = SkewEstimator()
        with self.assertRaises(ValueError):
            estimator.add_sample(float("nan"), 100.0)

    def test_add_sample_nan_local_raises(self):
        estimator = SkewEstimator()
        with self.assertRaises(ValueError):
            estimator.add_sample(100.0, float("nan"))

    def test_add_sample_inf_raises(self):
        estimator = SkewEstimator()
        with self.assertRaises(ValueError):
            estimator.add_sample(float("inf"), 100.0)
        with self.assertRaises(ValueError):
            estimator.add_sample(100.0, float("-inf"))

    def test_negative_timestamps_allowed(self):
        # Some systems represent time as a signed offset.
        estimator = SkewEstimator()
        estimator.add_sample(-5.0, -10.0)
        result = estimator.estimate()
        self.assertEqual(result.skew_seconds, 5.0)

    def test_reset_clears_samples(self):
        estimator = SkewEstimator()
        estimator.add_sample(110.0, 100.0)
        estimator.add_sample(111.0, 101.0)
        estimator.reset()
        with self.assertRaises(ValueError):
            estimator.estimate()

    def test_result_is_frozen(self):
        estimator = SkewEstimator()
        estimator.add_sample(101.0, 100.0)
        result = estimator.estimate()
        with self.assertRaises(Exception):
            result.skew_seconds = 999.0

    def test_float_return_type(self):
        estimator = SkewEstimator()
        estimator.add_sample(100.5, 100.0)
        result = estimator.estimate()
        self.assertIsInstance(result.skew_seconds, float)
        self.assertIsInstance(result.median_absolute_deviation, float)

    def test_single_sample(self):
        estimator = SkewEstimator()
        estimator.add_sample(123.0, 100.0)
        result = estimator.estimate()
        self.assertEqual(result.skew_seconds, 23.0)
        self.assertEqual(result.sample_count, 1)
        self.assertEqual(result.median_absolute_deviation, 0.0)


if __name__ == "__main__":
    unittest.main()
