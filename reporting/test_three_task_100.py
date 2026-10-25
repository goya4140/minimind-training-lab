import unittest
from collections import Counter
from three_task_100 import text_cases,first_pair


class ThreeTaskTests(unittest.TestCase):
    def test_text_cases_fixed_balanced_unique(self):
        cases=text_cases();self.assertEqual(len(cases),100);self.assertEqual(len({c['id'] for c in cases}),100)
        self.assertEqual(set(Counter(c['category'] for c in cases).values()),{20})
        self.assertEqual(cases,text_cases());self.assertEqual(sum(c['strict_format'] for c in cases),20)

    def test_pair_is_original_not_tool_instruction(self):
        self.assertEqual(first_pair('[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]'),('q','a'))
        with self.assertRaises(ValueError):first_pair([{'role':'tool','content':'untrusted'}])


if __name__=='__main__':unittest.main()
