import os
import json
import pandas as pd
from tqdm import tqdm
import sys
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stepmania_difficulty_predictor.features.HorizontalDensity import HorizontalDensity
from stepmania_difficulty_predictor.features.VerticalDensity import VerticalDensity
from stepmania_difficulty_predictor.features.StreamDetector import StreamDetector
from stepmania_difficulty_predictor.features.PatternDetector import PatternDetector

def build_features(processed_dir, output_path):
    """
    Builds a feature set from the processed chart files and saves it to a CSV.
    """
    chart_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith('.chart')]

    if not chart_files:
        print(f"No .chart files found in {processed_dir}. Did you run make_dataset_from_sm.py first?")
        return

    # Initialize feature extractors
    horizontal_density = HorizontalDensity(alpha=3)
    vertical_density = VerticalDensity(alpha=3)
    stream_detector = StreamDetector()
    pattern_detector = PatternDetector()

    all_features = []

    print("Building features from processed chart files...")
    for chart_file in tqdm(chart_files):
        try:
            with open(chart_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Skipping corrupt chart file: {chart_file} ({e})")
            continue

        chart = data.get('chart', {})
        if chart:
            chart = {float(k): v for k, v in chart.items()}

        mode = data.get('mode', 'unknown')
        meter = data.get('meter', 0)

        if not chart:
            continue

        # Compute features using the mode-agnostic extractors
        h_density_features = horizontal_density.compute(chart)
        v_density_features = vertical_density.compute(chart)
        stream_features = stream_detector.compute(chart)
        pattern_features = pattern_detector.compute(chart)

        # Combine all features into a single dictionary
        features = {
            'meter': meter,
            'mode': mode,
            **h_density_features,
            **v_density_features,
            **stream_features,
            **pattern_features
        }
        all_features.append(features)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(all_features)
    df.to_csv(output_path, index=False)
    print(f"Successfully built feature dataset with {len(df)} charts at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build features from processed chart files.")
    parser.add_argument("processed_dir", type=str, help="Directory containing the processed .chart files.")
    parser.add_argument("output_path", type=str, help="Path to save the output dataset.csv file.")
    args = parser.parse_args()
    build_features(args.processed_dir, args.output_path)
