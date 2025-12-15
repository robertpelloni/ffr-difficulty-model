import unittest
import os
import simfile
import pickle
from stepmania_difficulty_predictor.models.prediction_pipeline import ModeAgnosticDifficultyPredictor

class MockModel:
    """A mock model for testing purposes."""
    def __init__(self, prediction_value=1.0):
        self.prediction_value = prediction_value
        self.feature_names_in_ = ['nps', 'length', 'col_0', 'col_1', 'col_2', 'col_3',
                                  'left', 'right', 'all', 'stream_percentage',
                                  'max_stream_length', 'jack_percentage', 'crossover_percentage']

    def predict(self, features):
        return [self.prediction_value]

class TestModeAgnosticDifficultyPredictor(unittest.TestCase):

    def setUp(self):
        self.model_dir = "stepmania_difficulty_predictor/model"
        os.makedirs(self.model_dir, exist_ok=True)

        # Create valid, empty pickle files for the dummy models
        with open(os.path.join(self.model_dir, "dance-single.p"), "wb") as f:
            pickle.dump(MockModel(), f)
        with open(os.path.join(self.model_dir, "dance-double.p"), "wb") as f:
            pickle.dump(MockModel(), f)

        self.sm_path = "test.sm"
        self.dance_double_path = "tests/dance_double.sm"
        self.empty_chart_path = "tests/empty_chart.sm"
        self.predictor = ModeAgnosticDifficultyPredictor(model_dir=self.model_dir)

    def test_predict_from_path(self):
        """
        Tests that the predictor can successfully predict the difficulty of a .sm file.
        """
        self.predictor.models['dance-single'] = MockModel(prediction_value=1.0)

        predictions = self.predictor.predict(self.sm_path)
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

        for p in predictions:
            self.assertEqual(p['mode'], 'dance-single')
            self.assertEqual(p['predicted_difficulty'], 1.0)

    def test_predict_from_object(self):
        """
        Tests that the predictor can successfully predict from a simfile object.
        """
        self.predictor.models['dance-single'] = MockModel(prediction_value=2.0)

        with open(self.sm_path, "r") as f:
            sm = simfile.load(f)
            predictions = self.predictor.predict(sm)
            self.assertIsInstance(predictions, list)
            self.assertGreater(len(predictions), 0)
            self.assertEqual(predictions[0]['predicted_difficulty'], 2.0)

    def test_predict_batch(self):
        """
        Tests batch prediction functionality.
        """
        self.predictor.models['dance-single'] = MockModel(prediction_value=3.0)

        batch_predictions = self.predictor.predict_batch([self.sm_path, self.sm_path])
        self.assertIsInstance(batch_predictions, list)
        self.assertEqual(len(batch_predictions), 2)
        self.assertEqual(batch_predictions[0][0]['predicted_difficulty'], 3.0)

    def test_include_features(self):
        """
        Tests that the feature vector is correctly included in the output.
        """
        self.predictor.models['dance-single'] = MockModel()

        predictions = self.predictor.predict(self.sm_path, include_features=True)
        self.assertGreater(len(predictions), 0)
        self.assertIn('features', predictions[0])
        self.assertIsInstance(predictions[0]['features'], dict)

    def test_handles_other_modes(self):
        """
        Tests that the predictor correctly processes a file with a different mode.
        """
        self.predictor.models['dance-double'] = MockModel(prediction_value=4.0)

        predictions = self.predictor.predict(self.dance_double_path)
        self.assertGreater(len(predictions), 0)
        self.assertEqual(predictions[0]['mode'], 'dance-double')
        self.assertEqual(predictions[0]['predicted_difficulty'], 4.0)

    def test_empty_chart(self):
        """
        Tests graceful handling of charts with no notes.
        """
        predictions = self.predictor.predict(self.empty_chart_path)
        self.assertEqual(predictions, [])

    def test_no_model_for_mode(self):
        """
        Tests that charts are skipped when no corresponding model is loaded.
        """
        if 'dance-single' in self.predictor.models:
            del self.predictor.models['dance-single']

        predictions = self.predictor.predict(self.sm_path)
        self.assertEqual(predictions, [])

if __name__ == '__main__':
    unittest.main()
