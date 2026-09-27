import unittest

from nl_pl_kgc.data import load_examples


class DataTests(unittest.TestCase):
    def test_load_examples(self):
        examples = load_examples("tests/fixtures.json")
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].triples[0], ("nausea", "Adverse_effect", "DrugX"))
        self.assertEqual(examples[1].triples, ())
