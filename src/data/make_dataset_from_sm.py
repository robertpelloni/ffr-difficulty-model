# -*- coding: utf-8 -*-
import os
import sys
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
    processed_files = 0
    for sm_file in simfiles:
        try:
            preprocessed_charts = preprocessor.preprocess(sm_file)
            for chart_data in preprocessed_charts:
                serializer.download(chart_data, chart_id)
                chart_id += 1
            processed_files += 1
            print(f"Processed {sm_file.title} ({processed_files}/{len(simfiles)})")
        except Exception as e:
            print(f"Error processing {sm_file.title}: {e}", file=sys.stderr)

    print(f"Processed and serialized {chart_id} charts from {processed_files} files.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help='Input folder containing .sm files')
    parser.add_argument('output_folder', type=str, help='Output folder for .chart files')
    args = parser.parse_args()

    main(args.input_folder, args.output_folder)
