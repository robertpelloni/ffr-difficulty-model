import unittest
import os
from src.models.prediction_pipeline import DifficultyPredictor

class TestDifficultyPredictor(unittest.TestCase):

    def setUp(self):
        self.model_path = "models/random_forest_regressor.p"
        self.sm_path = "test.sm"
        self.predictor = DifficultyPredictor(self.model_path)

    def test_predict_difficulty(self):
        """
        Tests that the DifficultyPredictor can successfully predict the difficulty
        of a .sm file and that the prediction is a float.
        """
        predictions = self.predictor.predict(self.sm_path)
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

        for p in predictions:
            self.assertIsInstance(p['predicted_difficulty'], float)

if __name__ == '__main__':
    unittest.main()
