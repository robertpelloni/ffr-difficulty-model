import os
import dotenv
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import ast

if __name__ == '__main__':

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

    processed_data_folder = os.getenv("PROCESSED_DATA_FOLDER", "data/processed")
    models_folder = os.getenv("MODELS_FOLDER", "models")
    os.makedirs(models_folder, exist_ok=True)

    df = pd.read_csv(os.path.join(processed_data_folder, 'dataset.csv'), index_col='id')

    def safe_literal_eval(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return x
        return x

    # Convert string representations of dictionaries into actual dictionaries
    df['vertical'] = df['vertical'].apply(safe_literal_eval)
    df['horizontal'] = df['horizontal'].apply(safe_literal_eval)

    # Flatten the dictionary columns
    df = pd.concat([df.drop(['vertical', 'horizontal'], axis=1),
                   df['vertical'].apply(pd.Series),
                   df['horizontal'].apply(pd.Series)], axis=1)

    # Map difficulty strings to numerical values
    difficulty_mapping = {
        'Beginner': 1,
        'Easy': 2,
        'Medium': 3,
        'Hard': 4,
        'Challenge': 5,
        'Edit': 0 # Assuming 'Edit' difficulty is not for training
    }
    df['difficulty'] = df['difficulty'].map(difficulty_mapping)
    df = df[df.difficulty != 0]

    y_true = df.pop('difficulty')

    # Drop non-numeric and unnecessary columns
    df = df.select_dtypes(include=['number'])

    X_train, X_test, y_train, y_test = train_test_split(df, y_true, random_state=0)

    regr = RandomForestRegressor(random_state=0)
    regr.fit(X_train, y_train)

    print(f"Random Forest Regressor Score: {regr.score(X_test, y_test)}")

    with open(os.path.join(models_folder, 'random_forest_regressor.p'), "wb") as f:
        pickle.dump(regr, f)
