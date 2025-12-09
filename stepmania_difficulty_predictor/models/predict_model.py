import os
import dotenv
import pickle
import pandas as pd
import ast

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

    processed_data_folder = os.getenv("PROCESSED_DATA_FOLDER", "data/processed")
    models_folder = os.getenv("MODELS_FOLDER", "models")
    output_data_folder = os.getenv("OUTPUT_DATA_FOLDER", "data/output")
    os.makedirs(output_data_folder, exist_ok=True)

    # Load the dataset
    df = pd.read_csv(os.path.join(processed_data_folder, 'dataset.csv'), index_col='id')

    def safe_literal_eval(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return x
        return x

    # Preprocess the data just like in the training script
    df['vertical'] = df['vertical'].apply(safe_literal_eval)
    df['horizontal'] = df['horizontal'].apply(safe_literal_eval)

    df_features = pd.concat([df.drop(['vertical', 'horizontal', 'difficulty'], axis=1),
                             df['vertical'].apply(pd.Series),
                             df['horizontal'].apply(pd.Series)], axis=1)

    # Map difficulty strings to numerical values
    difficulty_mapping = {
        'Beginner': 1, 'Easy': 2, 'Medium': 3, 'Hard': 4, 'Challenge': 5, 'Edit': 0
    }
    df['difficulty'] = df['difficulty'].map(difficulty_mapping)
    y_true = df['difficulty']

    # Ensure all feature columns are numeric
    df_features = df_features.select_dtypes(include=['number'])

    # Load the trained model
    with open(os.path.join(models_folder, 'random_forest_regressor.p'), "rb") as f:
        regr = pickle.load(f)

    # Generate predictions
    predictions = regr.predict(df_features)

    # Create a results DataFrame
    results = pd.DataFrame({
        'actual_difficulty': y_true,
        'predicted_difficulty': predictions
    }, index=df.index)

    # Save the results
    results.to_csv(os.path.join(output_data_folder, 'predictions.csv'))

    print("Predictions saved to data/output/predictions.csv")
