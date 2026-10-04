import unittest

from nl_pl_kgc.train import _sequence_length_kwargs


class TrainingCompatibilityTests(unittest.TestCase):
    def test_legacy_trl_sequence_length_name(self):
        class LegacySFTConfig:
            def __init__(self, output_dir, max_seq_length=None):
                pass

        self.assertEqual(
            _sequence_length_kwargs(LegacySFTConfig, 2048),
            {"max_seq_length": 2048},
        )

    def test_current_trl_sequence_length_name(self):
        class CurrentSFTConfig:
            def __init__(self, output_dir, max_length=None):
                pass

        self.assertEqual(
            _sequence_length_kwargs(CurrentSFTConfig, 2048),
            {"max_length": 2048},
        )

