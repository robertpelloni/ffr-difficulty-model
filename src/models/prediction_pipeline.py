import pickle
import pandas as pd
import ast
import simfile
from src.data.SMChartPreprocessor import SMChartPreprocessor
from src.features.HorizontalDensity import HorizontalDensity
from src.features.VerticalDensity import VerticalDensity

def predict_difficulty(sm_path: str, model_path: str) -> float:
    """
    Predicts the difficulty of a single .sm file.

    Args:
        sm_path: The path to the .sm file.
        model_path: The path to the trained model.

    Returns:
        The predicted difficulty.
    """
    # 1. Load and preprocess the .sm file
    sm_file = simfile.open(sm_path)
    preprocessor = SMChartPreprocessor()
    preprocessed_charts = preprocessor.preprocess(sm_file)

    if not preprocessed_charts:
        raise ValueError("No valid charts found in the .sm file.")

    # For simplicity, we'll just use the first chart
    chart_data = preprocessed_charts[0]

    # 2. Build features
    chart = chart_data.pop('chart')
    horizontal_density = HorizontalDensity(alpha=3)
    vertical_density = VerticalDensity(alpha=3)
    horizontal_features = horizontal_density.compute(chart)

    features = {
        'meter': chart_data.get('meter'),
        'nps': horizontal_features.get('nps'),
        'length': horizontal_features.get('length'),
    }
    vertical_features = vertical_density.compute(chart)

    # Flatten the dictionary features
    features.update(vertical_features)
    features.update(horizontal_features)

    df_features = pd.DataFrame([features])
    df_features = df_features.select_dtypes(include=['number'])

    # 3. Load the model and predict
    with open(model_path, "rb") as f:
        regr = pickle.load(f)

    prediction = regr.predict(df_features)[0]

    return prediction
