import unittest
from summarize_three_task_100 import summary_counts,review_map,card


class ReviewSummaryTests(unittest.TestCase):
    def test_uncertain_denominator_is_explicit(self):
        r=summary_counts(['correct','incorrect','uncertain']);self.assertEqual(r['correct_over_all'],1/3);self.assertEqual(r['correct_over_assessable'],.5)

    def test_unknown_and_missing_reviews_rejected(self):
        with self.assertRaises(ValueError):summary_counts(['pass'])
        with self.assertRaises(ValueError):review_map([{'id':'x','verdict':'correct','reason':'test'}],['x','y'])
        with self.assertRaises(ValueError):review_map([{'id':'x','verdict':'correct','reason':''}],['x'])

    def test_generated_html_escapes_untrusted_content(self):
        case={'id':'T001','task':'text','category':'fixture','prompt':'<script>bad</script>','reference':'<img>'}
        pred={'completion':'<script>bad</script>','completion_token_ids':[1],'stopped_at_eos':True}
        text=card(case,pred,{'verdict':'incorrect','reason':'<b>fixture</b>'})
        self.assertNotIn('<script>',text);self.assertIn('&lt;script&gt;',text)


if __name__=='__main__':unittest.main()
