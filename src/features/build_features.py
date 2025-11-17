import dotenv
import os
import pickle
import csv

from src.features.HorizontalDensity import HorizontalDensity
from src.features.VerticalDensity import VerticalDensity

if __name__ == '__main__':

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

    processed_data_folder = os.getenv("PROCESSED_DATA_FOLDER", "data/processed")
    os.makedirs(processed_data_folder, exist_ok = True)

    vertical_density = VerticalDensity(alpha=3)
    horizontal_density = HorizontalDensity(alpha=3)

    fields = ['id', 'difficulty', 'meter', 'nps', 'length', 'vertical', 'horizontal']

    with open(os.path.join(processed_data_folder, 'dataset.csv'), 'w') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for filename in os.listdir(processed_data_folder):
            if filename.endswith(".chart"):
                filepath = os.path.join(processed_data_folder, filename)
                with open(filepath, "rb") as chart_file:
                    raw_data = pickle.load(chart_file)

                chart = raw_data.pop('chart')
                horizontal_features = horizontal_density.compute(chart)

                row = {
                    'id': int(filename.split('.')[0]),
                    'difficulty': raw_data.get('difficulty'),
                    'meter': raw_data.get('meter'),
                    'nps': horizontal_features.get('nps'),
                    'length': horizontal_features.get('length'),
                    'vertical': vertical_density.compute(chart),
                    'horizontal': horizontal_features
                }
                w.writerow(row)
