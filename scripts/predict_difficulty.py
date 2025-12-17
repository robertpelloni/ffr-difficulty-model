import argparse
import sys
import os
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stepmania_difficulty_predictor.models.prediction_pipeline import ModeAgnosticDifficultyPredictor

def predict_difficulty_cli(file_path, model_dir=None, use_json=False):
    """
    Command-line interface for the difficulty predictor.
    """
    if model_dir:
        predictor = ModeAgnosticDifficultyPredictor(model_dir=model_dir)
    else:
        predictor = ModeAgnosticDifficultyPredictor()

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    predictions = predictor.predict(file_path)

    if use_json:
        print(json.dumps(predictions, indent=4))
    else:
        if not predictions:
            print(f"No charts could be processed for '{os.path.basename(file_path)}'.")
            print("This could be due to missing models for the chart modes or an invalid file.")
        else:
            print(f"Predictions for '{os.path.basename(file_path)}':")
            for p in predictions:
                print(
                    f"  - Mode: {p['mode']}, Difficulty: {p['difficulty']}, "
                    f"Meter: {p['meter']} -> Predicted Meter: {p['predicted_difficulty']:.2f}"
                )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict the difficulty of StepMania charts.")
    parser.add_argument("file_path", type=str, help="Path to the .sm file.")
    parser.add_argument("--model_dir", type=str, help="Path to the directory containing trained models.")
    parser.add_argument("--json", action="store_true", help="Output predictions in JSON format.")

    args = parser.parse_args()
    predict_difficulty_cli(args.file_path, args.model_dir, args.json)
