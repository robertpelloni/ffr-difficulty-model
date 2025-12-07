import os
import dotenv
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import re

def load_and_preprocess_data(data_path):
    """Loads and preprocesses the dataset from a CSV file."""
    df = pd.read_csv(data_path, index_col='id')

    def extract_features_from_string(x):
        if not isinstance(x, str):
            return {}

        # A robust regex to find all floating point numbers in the string
        numbers = [float(n) for n in re.findall(r"[-+]?\\d*\\.\\d+|\\d+", x)]

        if len(numbers) == 7: # vertical
            return {'L': numbers[0], 'D': numbers[1], 'U': numbers[2], 'R': numbers[3],
                    'left': numbers[4], 'right': numbers[5], 'all': numbers[6]}
        elif len(numbers) == 2: # horizontal
            return {'nps': numbers[0], 'length': numbers[1]}
        return {}

    # Apply the function to the 'vertical' and 'horizontal' columns
    vertical_df = df['vertical'].apply(extract_features_from_string).apply(pd.Series)
    horizontal_df = df['horizontal'].apply(extract_features_from_string).apply(pd.Series)

    # Concatenate the new columns, avoiding duplicates from the original df
    df = pd.concat([df.drop(['vertical', 'horizontal'], axis=1),
                   vertical_df,
                   horizontal_df], axis=1)

    # Remove duplicate columns, keeping the last one (from the dictionaries)
    df = df.loc[:,~df.columns.duplicated(keep='last')]

    return df

def train_model(df, models_folder):
    """Trains the model and saves it to a file."""
    # Map difficulty strings to numerical values
    difficulty_mapping = {
        'Beginner': 1, 'Easy': 2, 'Medium': 3,
        'Hard': 4, 'Challenge': 5, 'Edit': 0, 'Unknown': 0
    }
    df['difficulty'] = df['difficulty'].map(difficulty_mapping)
    df = df[df['difficulty'] != 0].dropna(subset=['difficulty'])

    y_true = df.pop('difficulty')

    # Define feature columns explicitly to ensure they are all included
    feature_cols = [
        'meter', 'nps', 'length', 'L', 'D', 'U', 'R', 'left', 'right', 'all'
    ]

    # Ensure all feature columns are present, filling missing ones with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y_true, random_state=0)

    regr = RandomForestRegressor(random_state=0)
    regr.fit(X_train, y_train)

    print(f"Random Forest Regressor Score: {regr.score(X_test, y_test)}")

    with open(os.path.join(models_folder, 'random_forest_regressor.p'), "wb") as f:
        pickle.dump(regr, f)

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

    processed_data_folder = os.getenv("PROCESSED_DATA_FOLDER", "data/processed")
    models_folder = os.getenv("MODELS_FOLDER", "models")
    os.makedirs(models_folder, exist_ok=True)

    dataset_path = os.path.join(processed_data_folder, 'dataset.csv')

    df = load_and_preprocess_data(dataset_path)
    train_model(df, models_folder)
