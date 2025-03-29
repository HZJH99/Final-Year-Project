import unittest
from app import preprocess_text, classify_sentiment

class TestSentimentFunctions(unittest.TestCase):
    
    def test_preprocess_text_lowercase(self):
        """Test if text is converted to lowercase"""
        text = "This is a Test!"
        cleaned = preprocess_text(text, skip_language_detection=True)
        self.assertTrue(cleaned.islower(), "Text has to be converted to lowercase")

    def test_preprocess_text_removes_special_chars(self):
        """Test if special characters are removed from text"""
        text = "The food that I ordered was great! :)"
        cleaned = preprocess_text(text, skip_language_detection=True)
        self.assertNotIn("!", cleaned, "Exclamation mark should be removed")
        self.assertNotIn(":", cleaned, "Colon should be removed")   
        self.assertNotIn(")", cleaned, "Closing parenthesis should be removed")

    def test_preprocess_text_negation_handling(self):
        """Test if negation is preserved as 'not'"""
        text = "I don't like this product"
        cleaned = preprocess_text(text, skip_language_detection=True)
        self.assertIn("not", cleaned, "Negation should be preserved as not_")

    def test_classify_sentiment_positive(self):
        """Test classify_sentiment for positive score"""
        score = 0.8
        self.assertEqual(classify_sentiment(score), "Positive")

    def test_classify_sentiment_neutral(self):
        """Test classify_sentiment for neutral score"""
        score = 0.5
        self.assertEqual(classify_sentiment(score), "Neutral")

    def test_classify_sentiment_negative(self):
        """Test classify_sentiment for negative score"""
        score = 0.2
        self.assertEqual(classify_sentiment(score), "Negative")

    def test_preprocess_text_skips_language_detection(self):
        """Test if language detection is skipped when flag is True"""
        text = "Un biglietto, per favore"
        cleaned = preprocess_text(text, skip_language_detection=True)
        self.assertTrue(cleaned, str)

    def test_preprocess_text_removes_numbers_and_keeps_spaces(self):
        """Test if numbers are retained and special characters removed"""
        text = "I have 3 children?"
        cleaned = preprocess_text(text, skip_language_detection=True)
        self.assertIn("3", cleaned)
        self.assertNotIn("?", cleaned)
        self.assertTrue(" " in cleaned)

    def test_preprocess_text_empty_string(self):
        """Test preprocessing with an empty string"""
        cleaned = preprocess_text("", skip_language_detection=True)
        self.assertEqual(cleaned, "")

    def test_classify_sentiment_exact_thresholds(self):
        """Test classify_sentiment for exact threshold values"""
        self.assertEqual(classify_sentiment(0.7), "Positive")
        self.assertEqual(classify_sentiment(0.59), "Neutral")
        self.assertEqual(classify_sentiment(0.5), "Neutral")
        self.assertEqual(classify_sentiment(0.39), "Negative")

    def test_foreign_language_skip_translation(self):
        """Test foreign language input with translation disabled"""
        text = "我喜欢喝牛奶"
        cleaned = preprocess_text(text, skip_language_detection=True)
        self.assertIsInstance(cleaned, str)


if __name__ == '__main__':
    unittest.main()