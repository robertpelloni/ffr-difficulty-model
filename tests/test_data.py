import unittest
import pandas as pd
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from scripts.train_model import load_and_preprocess_data

class TestData(unittest.TestCase):

    def setUp(self):
        self.data_path = "data/processed/dataset.csv"

    def test_load_and_preprocess_data(self):
        """
        Tests that the data loading and preprocessing function returns a
        pandas DataFrame with the expected columns.
        """
        df = load_and_preprocess_data(self.data_path)
        self.assertIsInstance(df, pd.DataFrame)

        expected_cols = [
            'difficulty', 'meter', 'nps', 'length', 'L', 'D', 'U',
            'R', 'left', 'right', 'all'
        ]

        for col in expected_cols:
            self.assertIn(col, df.columns)

if __name__ == '__main__':
    unittest.main()
