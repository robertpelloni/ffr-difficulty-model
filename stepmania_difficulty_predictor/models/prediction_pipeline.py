import pickle
import pandas as pd
import simfile
from stepmania_difficulty_predictor.data.SMChartPreprocessor import SMChartPreprocessor
from stepmania_difficulty_predictor.features.HorizontalDensity import HorizontalDensity
from stepmania_difficulty_predictor.features.VerticalDensity import VerticalDensity
from stepmania_difficulty_predictor.features.StreamDetector import StreamDetector
from stepmania_difficulty_predictor.features.PatternDetector import PatternDetector
from typing import Union, List
import os

# Get the path to the packaged model
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'random_forest_regressor.p')

class DifficultyPredictor:
    """
    A class to predict the difficulty of StepMania (.sm) files.

    This class provides a high-level interface for predicting the difficulty of
    StepMania charts. It can be initialized with a path to a trained model,
    and can make predictions on either a file path or a `simfile` object.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """
        Initializes the DifficultyPredictor.

        This method loads the trained model from the specified path and initializes
        the necessary preprocessors and feature extractors.

        Args:
            model_path: The path to the trained model file. If not provided,
                        the default packaged model will be used.
        """
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.preprocessor = SMChartPreprocessor()
        self.horizontal_density = HorizontalDensity(alpha=3)
        self.vertical_density = VerticalDensity(alpha=3)
        self.stream_detector = StreamDetector()
        self.pattern_detector = PatternDetector()

    def predict(self, sm: Union[str, simfile.Simfile], include_features: bool = False) -> list:
        """
        Predicts the difficulty of all charts in a .sm file or simfile object.

        This method takes either a path to a .sm file or a `simfile` object
        and returns a list of predictions for each valid chart in the file.

        Args:
            sm: The path to the .sm file or a `simfile` object.
            include_features: If True, the raw feature vector will be included
                              in the output.

        Returns:
            A list of dictionaries, where each dictionary contains the chart's
            original difficulty, its meter, and the predicted difficulty.
            If `include_features` is True, the dictionary will also contain a
            'features' key with the raw feature vector.
            Returns an empty list if no valid charts are found.
        """
        return self.predict_batch([sm], include_features=include_features)[0]

    def predict_batch(self, sms: List[Union[str, simfile.Simfile]], include_features: bool = False) -> List[list]:
        """
        Predicts the difficulty of all charts in a batch of .sm files or simfile objects.

        This method takes a list of paths to .sm files or a list of `simfile` objects
        and returns a list of prediction lists for each valid file.

        Args:
            sms: A list of paths to .sm files or a list of `simfile` objects.
            include_features: If True, the raw feature vector will be included
                              in the output.

        Returns:
            A list of lists of dictionaries, where each inner list contains the predictions
            for a single file. Each dictionary contains the chart's original difficulty,
            its meter, and the predicted difficulty. If `include_features` is True,
            the dictionary will also contain a 'features' key with the raw feature vector.
        """
        batch_predictions = []
        for sm in sms:
            try:
                if isinstance(sm, str):
                        sm_file = simfile.open(sm, strict=False)
                else:
                    sm_file = sm
            except ValueError:
                batch_predictions.append([])
                continue

            preprocessed_charts = self.preprocessor.preprocess(sm_file)

            if not preprocessed_charts:
                batch_predictions.append([])
                continue

            chart_predictions = []
            for chart_data in preprocessed_charts:
                chart = chart_data.pop('chart')

                horizontal_features = self.horizontal_density.compute(chart)
                vertical_features = self.vertical_density.compute(chart)
                stream_features = self.stream_detector.compute(chart)
                pattern_features = self.pattern_detector.compute(chart)

                features = {
                    'meter': chart_data.get('meter'),
                    **horizontal_features,
                    **vertical_features,
                    **stream_features,
                    **pattern_features

                    'nps': horizontal_features.get('nps'),
                    'length': horizontal_features.get('length')
 
                }

                df_features = pd.DataFrame([features])
                df_features = df_features.select_dtypes(include=['number'])

                # Ensure the order of columns matches the training order
                training_cols = [
                    'meter', 'nps', 'length', 'L', 'D', 'U', 'R', 'left', 'right', 'all',
                    'stream_percentage', 'max_stream_length', 'jack_percentage', 'crossover_percentage'
                ]
                df_features = df_features.reindex(columns=training_cols, fill_value=0)

                prediction = self.model.predict(df_features)[0]

                result = {
                    'difficulty': chart_data.get('difficulty'),
                    'meter': chart_data.get('meter'),
                    'predicted_difficulty': prediction
                }

                if include_features:
                    result['features'] = df_features.to_dict('records')[0]

                chart_predictions.append(result)

            batch_predictions.append(chart_predictions)

        return batch_predictions

def predict_difficulty(sm_path: str, model_path: str = DEFAULT_MODEL_PATH) -> float:
    """
    [DEPRECATED] Predicts the difficulty of a single .sm file.

    This function is provided for backward compatibility. For new applications,
    it is recommended to use the `DifficultyPredictor` class directly, as it
    is more efficient and provides more detailed output.
    """
    predictor = DifficultyPredictor(model_path)
    predictions = predictor.predict(sm_path)

    if not predictions:
        raise ValueError("No valid charts found in the .sm file.")

    return predictions[0]['predicted_difficulty']
