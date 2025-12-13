import dotenv
import os
import pickle
import csv
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from stepmania_difficulty_predictor.features.HorizontalDensity import HorizontalDensity
from stepmania_difficulty_predictor.features.VerticalDensity import VerticalDensity

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

    processed_data_folder = os.getenv("PROCESSED_DATA_FOLDER", "data/processed")
    os.makedirs(processed_data_folder, exist_ok = True)

    vertical_density = VerticalDensity(alpha=3)
    horizontal_density = HorizontalDensity(alpha=3)

    # Define the full, flattened feature set for the CSV header
    fields = [
        'id', 'difficulty', 'meter', 'nps', 'length',
        'L', 'D', 'U', 'R', 'left', 'right', 'all'
    ]

    with open(os.path.join(processed_data_folder, 'dataset.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for filename in os.listdir(processed_data_folder):
            if filename.endswith(".chart"):
                filepath = os.path.join(processed_data_folder, filename)
                with open(filepath, "rb") as chart_file:
                    raw_data = pickle.load(chart_file)

                chart = raw_data.pop('chart')

                # Compute features
                horizontal_features = horizontal_density.compute(chart)
                vertical_features = vertical_density.compute(chart)

                # Create a single row with all features flattened
                row = {
                    'id': int(filename.split('.')[0]),
                    'difficulty': raw_data.get('difficulty'),
                    'meter': raw_data.get('meter'),
                    'nps': horizontal_features.get('nps'),
                    'length': horizontal_features.get('length'),
                    **vertical_features,
                    **horizontal_features
                }

                # Ensure only the defined fields are written to the CSV
                filtered_row = {key: row[key] for key in fields if key in row}
                writer.writerow(filtered_row)
