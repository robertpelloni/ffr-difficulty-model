import argparse
import os
from src.models.prediction_pipeline import DifficultyPredictor

def main():
    parser = argparse.ArgumentParser(description="Predict the difficulty of a StepMania (.sm) file.")
    parser.add_argument("sm_path", type=str, help="The path to the .sm file.")
    parser.add_argument("--model-path", type=str, default="models/random_forest_regressor.p", help="The path to the trained model.")
    args = parser.parse_args()

    if not os.path.exists(args.sm_path):
        print(f"Error: File not found at {args.sm_path}")
        return

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    try:
        predictor = DifficultyPredictor(args.model_path)
        predictions = predictor.predict(args.sm_path)

        if not predictions:
            print("No valid charts found in the .sm file.")
            return

        for p in predictions:
            print(f"Difficulty: {p['difficulty']}, Meter: {p['meter']}, Predicted Difficulty: {p['predicted_difficulty']:.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
