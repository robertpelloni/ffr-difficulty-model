import argparse
import os
from src.models.prediction_pipeline import predict_difficulty

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
        difficulty = predict_difficulty(args.sm_path, args.model_path)
        print(difficulty)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
