# -*- coding: utf-8 -*-
import os
import dotenv
from sm_data_loader import load_sm_files_from_directory
from SMChartPreprocessor import SMChartPreprocessor
from DataSerializer import DataSerializer

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    os.makedirs(output_filepath, exist_ok=True)

    simfiles = load_sm_files_from_directory(input_filepath)
    preprocessor = SMChartPreprocessor()
    serializer = DataSerializer(folder=output_filepath)

    chart_id = 0
    for sm_file in simfiles:
        preprocessed_charts = preprocessor.preprocess(sm_file)
        for chart_data in preprocessed_charts:
            serializer.download(chart_data, chart_id)
            chart_id += 1

    print(f"Processed and serialized {chart_id} charts.")

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

    # For now, we'll use the .env variables if they exist, otherwise, default to data/raw and data/processed
    raw_data_folder = os.getenv("RAW_DATA_FOLDER", "data/raw")
    processed_data_folder = os.getenv("PROCESSED_DATA_FOLDER", "data/processed")

    main(raw_data_folder, processed_data_folder)
