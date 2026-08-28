from unittest import TestCase

from miner.analyze import effective_alpha


class CohortAlphaTests(TestCase):
    """DEC-CA-0012 cohort duels judge each challenger at alpha/k — the
    quantile, never the margin. A solo-alpha LCB overstates a cohort verdict."""

    def test_solo_duel_keeps_the_configured_alpha(self):
        self.assertEqual(effective_alpha(0.05, 1), 0.05)

    def test_cohort_tightens_the_quantile(self):
        self.assertAlmostEqual(effective_alpha(0.05, 3), 0.05 / 3)

    def test_degenerate_k_never_loosens(self):
        self.assertEqual(effective_alpha(0.05, 0), 0.05)
