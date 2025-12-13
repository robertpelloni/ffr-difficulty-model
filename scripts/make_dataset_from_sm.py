# -*- coding: utf-8 -*-
import os
import sys
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from stepmania_difficulty_predictor.data.sm_data_loader import load_sm_files_from_directory
from stepmania_difficulty_predictor.data.SMChartPreprocessor import SMChartPreprocessor
from stepmania_difficulty_predictor.DataSerializer import DataSerializer

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    os.makedirs(output_filepath, exist_ok=True)

    simfiles = load_sm_files_from_directory(input_filepath)
    preprocessor = SMChartPreprocessor()
    serializer = DataSerializer(folder=output_filepath)

    chart_id = 0
    processed_files = 0
    for sm_file in simfiles:
        try:
            preprocessed_charts = preprocessor.preprocess(sm_file)
            for chart_data in preprocessed_charts:
                serializer.download(chart_data, chart_id)
                chart_id += 1
            processed_files += 1
        except Exception as e:
            print(f"Error processing {sm_file.title}: {e}", file=sys.stderr)

    print(f"Processed and serialized {chart_id} charts from {processed_files} files.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help='Input folder containing .sm files')
    parser.add_argument('output_folder', type=str, help='Output folder for .chart files')
    args = parser.parse_args()

    main(args.input_folder, args.output_folder)
