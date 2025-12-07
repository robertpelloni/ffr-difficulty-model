import pickle
import pandas as pd
import simfile
from src.data.SMChartPreprocessor import SMChartPreprocessor
from src.features.HorizontalDensity import HorizontalDensity
from src.features.VerticalDensity import VerticalDensity

class DifficultyPredictor:
    """
    Predicts the difficulty of StepMania (.sm) files.
    """
    def __init__(self, model_path: str):
        """
        Initializes the DifficultyPredictor by loading the trained model and preprocessors.

        Args:
            model_path: The path to the trained model file.
        """
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.preprocessor = SMChartPreprocessor()
        self.horizontal_density = HorizontalDensity(alpha=3)
        self.vertical_density = VerticalDensity(alpha=3)

    def predict(self, sm_path: str) -> list:
        """
        Predicts the difficulty of all charts in a single .sm file.

        Args:
            sm_path: The path to the .sm file.

        Returns:
            A list of dictionaries, where each dictionary contains the chart's
            difficulty, meter, and predicted difficulty.
        """
        sm_file = simfile.open(sm_path)
        preprocessed_charts = self.preprocessor.preprocess(sm_file)

        if not preprocessed_charts:
            return []

        predictions = []
        for chart_data in preprocessed_charts:
            chart = chart_data.pop('chart')

            horizontal_features = self.horizontal_density.compute(chart)

            features = {
                'meter': chart_data.get('meter'),
                'nps': horizontal_features.get('nps'),
                'length': horizontal_features.get('length'),
            }

            vertical_features = self.vertical_density.compute(chart)
            features.update(vertical_features)

            features.update(horizontal_features)

            df_features = pd.DataFrame([features])
            df_features = df_features.select_dtypes(include=['number'])

            prediction = self.model.predict(df_features)[0]

            predictions.append({
                'difficulty': chart_data.get('difficulty'),
                'meter': chart_data.get('meter'),
                'predicted_difficulty': prediction
            })

        return predictions

def predict_difficulty(sm_path: str, model_path: str) -> float:
    """
    [DEPRECATED] Predicts the difficulty of a single .sm file.
    Use the DifficultyPredictor class for better performance.

    This function now uses the DifficultyPredictor class internally but only
    returns the prediction for the first chart for backward compatibility.
    """
    predictor = DifficultyPredictor(model_path)
    predictions = predictor.predict(sm_path)

    if not predictions:
        raise ValueError("No valid charts found in the .sm file.")

    return predictions[0]['predicted_difficulty']
