import unittest

from nl_pl_kgc.parsing import parse_code_output, parse_natural_output


class ParsingTests(unittest.TestCase):
    def test_parse_natural_output(self):
        value = '[["nausea", "Adverse_effect", "DrugX"]]'
        self.assertEqual(parse_natural_output(value), [("nausea", "Adverse_effect", "DrugX")])

    def test_parse_code_output_without_execution(self):
        value = "return [Triple('nausea', 'Adverse_effect', 'DrugX')]"
        self.assertEqual(parse_code_output(value), [("nausea", "Adverse_effect", "DrugX")])

    def test_malicious_code_is_not_executed(self):
        value = "__import__('os').system('echo unsafe')"
        self.assertEqual(parse_code_output(value), [])
