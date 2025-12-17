import unittest
import numpy as np
from stepmania_difficulty_predictor.features.HorizontalDensity import HorizontalDensity
from stepmania_difficulty_predictor.features.VerticalDensity import VerticalDensity
from stepmania_difficulty_predictor.features.StreamDetector import StreamDetector
from stepmania_difficulty_predictor.features.PatternDetector import PatternDetector

class TestFeatures(unittest.TestCase):

    def setUp(self):
        # Sample chart data for testing
        self.chart = {
            1.0: "1001", 2.0: "0110", 3.0: "1100", 4.0: "0011",
            5.0: "1010", 6.0: "0101", 7.0: "1000", 8.0: "0100",
            9.0: "0010", 10.0: "0001", 10.05: "0001" # Jack
        }
        self.horizontal_density = HorizontalDensity(alpha=3)
        self.vertical_density = VerticalDensity(alpha=3)
        self.stream_detector = StreamDetector()
        self.pattern_detector = PatternDetector()

    def test_horizontal_density(self):
        """
        Tests that the HorizontalDensity class can successfully compute features.
        """
        features = self.horizontal_density.compute(self.chart)
        self.assertIsInstance(features, dict)
        self.assertIn('nps', features)
        self.assertIn('length', features)
        self.assertIsInstance(features['nps'], float)
        self.assertIsInstance(features['length'], float)

    def test_vertical_density(self):
        """
        Tests that the VerticalDensity class can successfully compute features.
        """
        features = self.vertical_density.compute(self.chart)
        self.assertIsInstance(features, dict)
        self.assertIn('L', features)
        self.assertIn('D', features)
        self.assertIn('U', features)
        self.assertIn('R', features)
        self.assertIn('left', features)
        self.assertIn('right', features)
        self.assertIn('all', features)
        self.assertIsInstance(features['L'], np.float64)
        self.assertIsInstance(features['D'], np.float64)
        self.assertIsInstance(features['U'], np.float64)
        self.assertIsInstance(features['R'], np.float64)

    def test_stream_detector(self):
        """
        Tests that the StreamDetector class can successfully compute features.
        """
        features = self.stream_detector.compute(self.chart)
        self.assertIsInstance(features, dict)
        self.assertIn('stream_percentage', features)
        self.assertIn('max_stream_length', features)
        self.assertIsInstance(features['stream_percentage'], float)
        self.assertIsInstance(features['max_stream_length'], int)

    def test_pattern_detector(self):
        """
        Tests that the PatternDetector class can successfully compute features.
        """
        features = self.pattern_detector.compute(self.chart)
        self.assertIsInstance(features, dict)
        self.assertIn('jack_percentage', features)
        self.assertIn('crossover_percentage', features)
        self.assertIsInstance(features['jack_percentage'], float)
        self.assertIsInstance(features['crossover_percentage'], float)
        self.assertGreater(features['jack_percentage'], 0) # Ensure jacks are detected

if __name__ == '__main__':
    unittest.main()
