import unittest
import json
import os
from app import app

class TestIntegration(unittest.TestCase):

    def setUp(self):
        # Set up Flask test client
        self.client = app.test_client()

    def test_home_page_access(self):
         """Test if home page loads successfully"""
         response = self.client.get('/')
         self.assertEqual(response.status_code, 200)
         self.assertIn(b"Sentiment Analysis", response.data)

    def test_upload_page_access(self):
        """Test if upload page loads successfully"""
        response = self.client.get('/upload_page')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upload CSV for Sentiment Analysis", response.data)

    def test_predict_single_review(self):
        """Test single review sentiment prediction"""
        sample_review = {"reviews": ["The food is beyond my imagination!"]}
        response = self.client.post(
            '/predict',
            data=json.dumps(sample_review),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('sentiment_scores', data)
        self.assertIn('sentiment_labels', data)
        self.assertEqual(len(data['sentiment_scores']), 1)
        self.assertEqual(len(data['sentiment_labels']), 1)

    def test_predict_empty_review(self):
        """Test empty review sentiment prediction"""
        sample_review = {"reviews": []}
        response = self.client.post(
            '/predict',
            data=json.dumps(sample_review),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"No reviews found in the request", response.data)

    def test_language_detection_api(self):
        """Test language detection API"""
        sample_text = {"text": "Es tut mir leid"}
        response = self.client.post(
            '/detect_language',
            data=json.dumps(sample_text),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('language', data)
    
    def test_upload_file_invalid(self):
         """Test upload route with no file"""
         response = self.client.post('/upload')
         self.assertEqual(response.status_code, 400)
         self.assertIn(b"No file uploaded", response.data)

    def test_upload_file_valid(self):
        """Test upload route with a valid CSV file"""
        test_csv_path = os.path.join('tests', 'sample_reviews.csv')

        # Create a simple CSV for testing
        with open(test_csv_path, 'w') as f:
            f.write("reviews\n")
            f.write("The food is delicious and yummy!\n")
            f.write("Worst service ever.\n")
        
        with open(test_csv_path, 'rb') as test_file:
            data = {'file': (test_file, 'sample_reviews.csv')}
            response = self.client.post('/upload', content_type='multipart/form-data', data=data)

        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertIn('overall_score', json_data)
        self.assertIn('overall_sentiment', json_data)
        self.assertIn('distribution', json_data)

        os.remove(test_csv_path) # Clean up

if __name__ == '__main__':
    unittest.main()