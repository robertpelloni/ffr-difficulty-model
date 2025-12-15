import os
import dotenv
import pickle
import pandas as pd
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

    processed_data_folder = os.getenv("PROCESSED_DATA_FOLDER", "data/processed")
    models_folder = "stepmania_difficulty_predictor/model"
    output_data_folder = os.getenv("OUTPUT_DATA_FOLDER", "data/output")
    os.makedirs(output_data_folder, exist_ok=True)

    # Load the dataset
    df = pd.read_csv(os.path.join(processed_data_folder, 'dataset.csv'), index_col='id')

    # Map difficulty strings to numerical values
    difficulty_mapping = {
        'Beginner': 1, 'Easy': 2, 'Medium': 3, 'Hard': 4, 'Challenge': 5, 'Edit': 0, 'Unknown': 0
    }
    df['difficulty'] = df['difficulty'].map(difficulty_mapping)
    y_true = df['difficulty']

    # Define feature columns explicitly to ensure they are all included
    feature_cols = [
        'meter', 'nps', 'length', 'L', 'D', 'U', 'R', 'left', 'right', 'all'
    ]

    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.fillna(0)

    # Load the trained model
    with open(os.path.join(models_folder, 'random_forest_regressor.p'), "rb") as f:
        regr = pickle.load(f)

    # Generate predictions
    predictions = regr.predict(X)

    # Create a results DataFrame
    results = pd.DataFrame({
        'actual_difficulty': y_true,
        'predicted_difficulty': predictions
    }, index=df.index)

    # Save the results
    results.to_csv(os.path.join(output_data_folder, 'predictions.csv'))

    print("Predictions saved to data/output/predictions.csv")
