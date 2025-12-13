import os
import dotenv
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import re
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def load_and_preprocess_data(data_path):
    """Loads and preprocesses the dataset from a CSV file."""
    df = pd.read_csv(data_path, index_col='id')
    return df

def get_features_and_labels(df):
    """Extracts features and labels from the DataFrame."""
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

    return X, y_true

def tune_and_train_model(X, y):
    """Tunes and trains the RandomForestRegressor model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    regr = RandomForestRegressor(random_state=0)
    grid_search = GridSearchCV(estimator=regr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_regr = grid_search.best_estimator_

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Random Forest Regressor Score with tuned parameters: {best_regr.score(X_test, y_test)}")

    return best_regr

def main():
    """Main function to load data, train the model, and save it."""
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

    processed_data_folder = os.getenv("PROCESSED_DATA_FOLDER", "data/processed")
    models_folder = "stepmania_difficulty_predictor/model"
    os.makedirs(models_folder, exist_ok=True)

    dataset_path = os.path.join(processed_data_folder, 'dataset.csv')

    df = load_and_preprocess_data(dataset_path)
    X, y = get_features_and_labels(df)
    model = tune_and_train_model(X, y)

    with open(os.path.join(models_folder, 'random_forest_regressor.p'), "wb") as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
