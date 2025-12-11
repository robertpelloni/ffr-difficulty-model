import unittest
import os
import simfile
from stepmania_difficulty_predictor.models.prediction_pipeline import DifficultyPredictor

class TestDifficultyPredictor(unittest.TestCase):

    def setUp(self):
        self.sm_path = "test.sm"
        self.predictor = DifficultyPredictor()

    def test_predict_from_path(self):
        """
        Tests that the DifficultyPredictor can successfully predict the difficulty
        of a .sm file from a path and that the prediction is a float.
        """
        predictions = self.predictor.predict(self.sm_path)
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

        for p in predictions:
            self.assertIsInstance(p['predicted_difficulty'], float)

    def test_predict_from_object(self):
        """
        Tests that the DifficultyPredictor can successfully predict the difficulty
        of a simfile object and that the prediction is a float.
        """
        with open(self.sm_path, "r") as f:
            sm = simfile.load(f)
            predictions = self.predictor.predict(sm)
            self.assertIsInstance(predictions, list)
            self.assertGreater(len(predictions), 0)

            for p in predictions:
                self.assertIsInstance(p['predicted_difficulty'], float)

    def test_custom_model_loading(self):
        """
        Tests that the DifficultyPredictor can successfully load a custom model.
        """
        custom_predictor = DifficultyPredictor(model_path="stepmania_difficulty_predictor/model/random_forest_regressor.p")
        predictions = custom_predictor.predict(self.sm_path)
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

    def test_predict_batch(self):
        """
        Tests that the DifficultyPredictor can successfully predict the difficulty
        of a batch of .sm files.
        """
        batch_predictions = self.predictor.predict_batch([self.sm_path, self.sm_path])
        self.assertIsInstance(batch_predictions, list)
        self.assertEqual(len(batch_predictions), 2)

        for predictions in batch_predictions:
            self.assertIsInstance(predictions, list)
            self.assertGreater(len(predictions), 0)

    def test_include_features(self):
        """
        Tests that the DifficultyPredictor can successfully include the feature
        vector in the prediction output.
        """
        predictions = self.predictor.predict(self.sm_path, include_features=True)
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

        for p in predictions:
            self.assertIn('features', p)
            self.assertIsInstance(p['features'], dict)

if __name__ == '__main__':
    unittest.main()
