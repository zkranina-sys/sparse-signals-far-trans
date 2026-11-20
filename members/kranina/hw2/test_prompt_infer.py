import unittest

import encoder as enc

OLD = 'unknown stays unknown; '
NEW = ('replace unknown fields with values directly implied by the asset name '
       '(sector, industry, region); examples not from the data: '
       '"Pacific Infrastructure Equity ETF" -> sector infrastructure, region Pacific; '
       '"Zeta Holdings S.A." -> nothing implied, stays unknown; '
       'keep unknown if the name gives no basis; ')
NOECHO = 'unknown stays unknown; keep every known field unchanged, and never echo the JSON; '


class PromptDifferenceTest(unittest.TestCase):
    def setUp(self):
        self.text = '{"name":"Gaslog Ltd.","sector":"unknown"}'
        self.para = enc.paraphrase_prompt(self.text)
        self.infer = enc.inference_prompt(self.text)
        self.noecho = enc.noecho_prompt(self.text)

    def test_infer_differs_by_one_fragment(self):
        self.assertEqual(self.para.count(OLD), 1)
        self.assertEqual(self.infer, self.para.replace(OLD, NEW, 1))
        self.assertNotEqual(self.para, self.infer)
        self.assertEqual(self.para.split(OLD, 1)[1], self.infer.split(NEW, 1)[1])

    def test_noecho_differs_by_one_fragment(self):
        self.assertEqual(self.noecho, self.para.replace(OLD, NOECHO, 1))


if __name__ == '__main__':
    unittest.main()
