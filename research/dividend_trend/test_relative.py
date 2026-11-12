import unittest

import numpy as np
import pandas as pd

from .relative import candidates, signal_targets


class RelativeTrendTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2020-01-01', periods=360)

    def test_registry(self):
        self.assertEqual(len(candidates()), 9)

    def test_blend_chooses_stronger_asset(self):
        prices = pd.DataFrame({'red': np.exp(np.arange(360) * .0005),
                               'gold': np.exp(np.arange(360) * .001)}, index=self.dates)
        result = signal_targets(prices, dict(family='dual', lookback='blend'))
        self.assertEqual(result.red.iloc[-1], 0)
        self.assertAlmostEqual(result.gold.iloc[-1], .95)

    def test_tie_prefers_red(self):
        prices = pd.DataFrame({'red': np.exp(np.arange(360) * .001),
                               'gold': np.exp(np.arange(360) * .001)}, index=self.dates)
        result = signal_targets(prices, dict(family='dual', lookback=126))
        self.assertAlmostEqual(result.red.iloc[-1], .95)
        self.assertEqual(result.gold.iloc[-1], 0)

    def test_both_negative_are_cash(self):
        prices = pd.DataFrame({'red': np.exp(-np.arange(360) * .001),
                               'gold': np.exp(-np.arange(360) * .002)}, index=self.dates)
        result = signal_targets(prices, dict(family='dual', lookback='blend'))
        self.assertEqual(result.to_numpy().sum(), 0)

    def test_core_is_explicit_even_when_satellite_is_cash(self):
        prices = pd.DataFrame({'red': np.exp(-np.arange(360) * .001),
                               'gold': np.exp(-np.arange(360) * .002)}, index=self.dates)
        result = signal_targets(prices, dict(family='core50', lookback=126))
        self.assertAlmostEqual(result.red.iloc[-1], .5)
        self.assertEqual(result.gold.iloc[-1], 0)

    def test_blend_does_not_use_future(self):
        prices = pd.DataFrame({'red': np.exp(np.arange(360) * .001),
                               'gold': np.exp(np.arange(360) * .0009)}, index=self.dates)
        candidate = dict(family='dual', lookback='blend')
        original = signal_targets(prices, candidate).iloc[:-1]
        prices.iloc[-1] = [1e6, 1e-6]
        pd.testing.assert_frame_equal(original, signal_targets(prices, candidate).iloc[:-1])


if __name__ == '__main__':
    unittest.main()
