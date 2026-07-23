import unittest

from miis_broadcast.core.models.livecc_transformers import LiveCCInfer


class LiveCCDegenerateFilterTests(unittest.TestCase):
    def classify(self, text: str) -> bool:
        classifier = object.__new__(LiveCCInfer)
        return classifier._is_degenerate(text, "Describe the current basketball action.")

    def test_normal_shot_is_not_rejected_for_function_words(self):
        self.assertFalse(self.classify("The player is shooting a basketball towards the hoop."))

    def test_normal_dribble_is_not_rejected_for_articles(self):
        self.assertFalse(self.classify("The player is dribbling the ball towards the basket."))

    def test_short_scoring_event_is_not_rejected(self):
        self.assertFalse(self.classify("The player shoots and scores."))

    def test_empty_output_is_rejected(self):
        self.assertTrue(self.classify(""))

    def test_numeric_stub_is_rejected(self):
        self.assertTrue(self.classify("5 ..."))

    def test_is_is_not_misread_as_first_person_i(self):
        self.assertFalse(self.classify("The player is dribbling the ball."))

    def test_youtube_style_hallucination_is_rejected(self):
        self.assertTrue(self.classify("Welcome back, please subscribe to the channel."))

    def test_first_person_narration_is_rejected(self):
        self.assertTrue(self.classify("I am dribbling and I am going to shoot the ball."))


if __name__ == "__main__":
    unittest.main()
