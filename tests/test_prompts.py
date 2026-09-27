import unittest

from nl_pl_kgc.data import Example
from nl_pl_kgc.prompts import build_prompt, format_triples


EXAMPLE = Example("DrugX caused nausea.", (("nausea", "Adverse_effect", "DrugX"),))


class PromptTests(unittest.TestCase):
    def test_natural_prompt_is_json_oriented(self):
        prompt = build_prompt(EXAMPLE.text, "natural", [EXAMPLE])
        self.assertIn("JSON array", prompt.system)
        self.assertIn('["nausea", "Adverse_effect", "DrugX"]', prompt.user)

    def test_code_prompt_uses_safe_literal_triples(self):
        prompt = build_prompt(EXAMPLE.text, "code", [EXAMPLE])
        self.assertIn("def extract", prompt.user)
        self.assertIn("Triple('nausea', 'Adverse_effect', 'DrugX')", prompt.user)
        self.assertEqual(format_triples([], "code"), "return []")
