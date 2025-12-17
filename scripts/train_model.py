import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import os
import argparse
import numpy as np

def train_model(dataset_path, model_dir):
    """
    Trains a separate model for each game mode in the dataset and saves them.
    """
    df = pd.read_csv(dataset_path)

    # Clean the data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Group by game mode and train a model for each
    for mode, group in df.groupby('mode'):
        print(f"--- Training model for mode: {mode} ---")

        if len(group) < 10:
            print(f"Skipping mode '{mode}': not enough data (found {len(group)} samples).")
            continue

        X = group.drop(columns=['meter', 'mode'])
        y = group['meter']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_leaf': [1, 2, 4]
        }

        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        print(f"Best model for '{mode}' has R^2 score: {r2:.3f}")

        # Save the trained model
        model_filename = f"{mode}.p"
        model_path = os.path.join(model_dir, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        print(f"Saved trained model for '{mode}' to {model_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a difficulty prediction model for each game mode.")
    parser.add_argument("dataset_path", type=str, help="Path to the feature dataset (dataset.csv).")
    parser.add_argument("model_dir", type=str, help="Directory to save the trained model files.")
    args = parser.parse_args()
    train_model(args.dataset_path, args.model_dir)
