from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .data import Triple


@dataclass(frozen=True)
class StrictMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    micro_f1: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "micro_f1": self.micro_f1,
        }


def strict_micro_metrics(
    gold: Sequence[Iterable[Triple]], predictions: Sequence[Iterable[Triple]]
) -> StrictMetrics:
    """Paper-aligned exact-match micro P/R/F1 over complete triples."""
    if len(gold) != len(predictions):
        raise ValueError("Gold and prediction collections must have equal length")
    tp = fp = fn = 0
    for expected, predicted in zip(gold, predictions):
        expected_set = set(expected)
        predicted_set = set(predicted)
        tp += len(expected_set & predicted_set)
        fp += len(predicted_set - expected_set)
        fn += len(expected_set - predicted_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return StrictMetrics(tp, fp, fn, precision, recall, f1)
