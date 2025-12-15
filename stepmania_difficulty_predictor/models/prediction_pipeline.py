import pickle
import pandas as pd
import simfile
import os
from typing import Union, List, Dict

# Add the project root to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stepmania_difficulty_predictor.data.SMChartPreprocessor import SMChartPreprocessor
from stepmania_difficulty_predictor.features.HorizontalDensity import HorizontalDensity
from stepmania_difficulty_predictor.features.VerticalDensity import VerticalDensity
from stepmania_difficulty_predictor.features.StreamDetector import StreamDetector
from stepmania_difficulty_predictor.features.PatternDetector import PatternDetector

# Get the path to the packaged models directory
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')

class ModeAgnosticDifficultyPredictor:
    """
    A class to predict the difficulty of StepMania (.sm) files for any game mode.

    This class provides a high-level interface for predicting chart difficulty.
    It automatically loads all available trained models and selects the appropriate
    one based on the chart's mode.
    """
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        """
        Initializes the ModeAgnosticDifficultyPredictor.

        This method scans the specified directory for model files (e.g., 'dance-single.p')
        and loads them into a dictionary.
        """
        self.models = self._load_models(model_dir)
        print(f"Loaded {len(self.models)} models for modes: {list(self.models.keys())}")

        self.preprocessor = SMChartPreprocessor()
        self.horizontal_density = HorizontalDensity(alpha=3)
        self.vertical_density = VerticalDensity(alpha=3)
        self.stream_detector = StreamDetector()
        self.pattern_detector = PatternDetector()

    def _load_models(self, model_dir: str) -> Dict[str, any]:
        """
        Scans a directory for .p files and loads them as models.
        """
        models = {}
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory not found at {model_dir}")
            return models

        for filename in os.listdir(model_dir):
            if filename.endswith('.p'):
                mode = filename.replace('.p', '')
                try:
                    with open(os.path.join(model_dir, filename), 'rb') as f:
                        models[mode] = pickle.load(f)
                except Exception as e:
                    print(f"Error loading model for mode '{mode}': {e}")
        return models

    def predict(self, sm: Union[str, simfile.Simfile], include_features: bool = False) -> list:
        """
        Predicts the difficulty of all charts in a .sm file or simfile object.
        """
        return self.predict_batch([sm], include_features=include_features)[0]

    def predict_batch(self, sms: List[Union[str, simfile.Simfile]], include_features: bool = False) -> List[list]:
        """
        Predicts the difficulty for a batch of .sm files or simfile objects.
        """
        batch_predictions = []
        for sm in sms:
            try:
                if isinstance(sm, str):
                    sm_file = simfile.open(sm, strict=False)
                else:
                    sm_file = sm
            except Exception as e:
                print(f"Error parsing simfile: {e}")
                batch_predictions.append([])
                continue

            preprocessed_charts = self.preprocessor.preprocess(sm_file)
            chart_predictions = []

            for chart_data in preprocessed_charts:
                mode = chart_data.get('mode')

                if mode not in self.models:
                    continue  # Skip modes for which we have no model

                chart = chart_data.get('chart', {})
                if not chart:
                    continue

                features = self._extract_features(chart, chart_data)

                df_features = pd.DataFrame([features])

                # Ensure the order of columns matches the training order, excluding mode
                training_cols = self.models[mode].feature_names_in_
                df_features = df_features.reindex(columns=training_cols, fill_value=0)

                prediction = self.models[mode].predict(df_features)[0]

                result = {
                    'mode': mode,
                    'difficulty': chart_data.get('difficulty'),
                    'meter': chart_data.get('meter'),
                    'predicted_difficulty': prediction
                }

                if include_features:
                    result['features'] = df_features.to_dict('records')[0]

                chart_predictions.append(result)

            batch_predictions.append(chart_predictions)

        return batch_predictions

    def _extract_features(self, chart: dict, chart_data: dict) -> dict:
        """
        Extracts a feature vector from a single chart.
        """
        h_density = self.horizontal_density.compute(chart)
        v_density = self.vertical_density.compute(chart)
        stream = self.stream_detector.compute(chart)
        pattern = self.pattern_detector.compute(chart)

        # We don't include meter or mode here as they are not features for the model
        return {**h_density, **v_density, **stream, **pattern}
