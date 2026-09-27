import unittest

from nl_pl_kgc.metrics import strict_micro_metrics


class MetricTests(unittest.TestCase):
    def test_strict_micro_f1(self):
        gold = [
            [("a", "r", "b"), ("c", "r", "d")],
            [("e", "s", "f")],
        ]
        predictions = [
            [("a", "r", "b"), ("wrong", "r", "d")],
            [("e", "s", "f")],
        ]
        result = strict_micro_metrics(gold, predictions)
        self.assertEqual(result.true_positives, 2)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 1)
        self.assertAlmostEqual(result.micro_f1, 2 / 3)
