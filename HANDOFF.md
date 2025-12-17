# Project Handoff: StepMania Difficulty Predictor

## 1. Project Overview

This document details the successful refactoring of the StepMania Difficulty Predictor from a collection of scripts into a robust, mode-agnostic, and reusable Python library. The final product is a powerful tool capable of predicting the difficulty of any StepMania (`.sm`) chart, for any game mode, making it suitable for integration into external tools like ArrowVortex and ddc.

The core of the library is the `ModeAgnosticDifficultyPredictor`, a class that automatically loads specialized, pre-trained models for each game mode and uses them to provide accurate difficulty predictions. The entire data pipeline, from raw `.sm` file processing to feature extraction and model training, has been generalized to support this multi-mode architecture.

## 2. Final Architecture

The project is structured as a standard, installable Python package with the following key components:

-   **`stepmania_difficulty_predictor/`**: The main package directory.
    -   **`data/`**: Contains the `SMChartPreprocessor`, which is responsible for parsing `.sm` files and converting them into a standardized, machine-readable format.
    -   **`features/`**: Contains the mode-agnostic feature extractors, including `HorizontalDensity`, `VerticalDensity`, `StreamDetector`, and `PatternDetector`.
    -   **`models/`**: Contains the `ModeAgnosticDifficultyPredictor` class, which is the primary interface for the library.
    -   **`model/`**: The directory where the trained model files (e.g., `dance-single.p`, `dance-double.p`) are stored.
-   **`scripts/`**: Contains the scripts for the data pipeline:
    -   `make_dataset_from_sm.py`: Processes raw `.sm` files into intermediate `.chart` files.
    -   `build_features.py`: Extracts features from the `.chart` files and generates the final `dataset.csv`.
    -   `train_model.py`: Trains a separate model for each game mode found in the dataset.
    -   `predict_difficulty.py`: A powerful command-line interface for the predictor.
-   **`tests/`**: Contains the unit tests for the project, ensuring the stability and correctness of the prediction pipeline.
-   **`setup.py`**: The package configuration file, which makes the library installable via `pip`.

## 3. Data and Model Training Pipeline

The end-to-end pipeline for training new models is as follows:

1.  **Place Raw Data**: Place all `.sm` files into the `data/raw` directory.
2.  **Process `.sm` Files**: Run `python scripts/make_dataset_from_sm.py data/raw data/processed` to convert the raw files into standardized `.chart` files.
3.  **Build Features**: Run `python scripts/build_features.py data/processed dataset.csv` to extract features from the `.chart` files and create the final `dataset.csv`.
4.  **Train Models**: Run `python scripts/train_model.py dataset.csv stepmania_difficulty_predictor/model` to train a separate model for each game mode and save them to the model directory.

## 4. Session History & Key Decisions

Our development journey was a comprehensive refactoring process with the following key milestones:

-   **Initial `.sm` Support**: We began by adapting the original, script-based predictor to work with `.sm` files, which involved creating a new data processing pipeline.
-   **Library Refactoring**: We then transformed the project into a proper Python library, introducing the `DifficultyPredictor` class, a `setup.py` file, and a command-line interface.
-   **Bug Fixing**: We identified and fixed several critical bugs, including a feature mismatch between the training and prediction pipelines, and numerous issues related to file encodings and data serialization formats.
-   **Mode-Agnostic Architecture**: The most significant enhancement was the generalization of the entire project to be mode-agnostic. This involved:
    -   Refactoring all feature extractors to dynamically adapt to the number of panels in a chart.
    -   Updating the data pipeline to produce a separate, specialized model for each game mode.
    -   Creating the `ModeAgnosticDifficultyPredictor` to automatically select the correct model for a given chart.
-   **Test Suite Overhaul**: We completely overhauled the test suite to match the new architecture, introducing a robust `MockModel` and ensuring the stability of the final product.

## 5. Key Learnings and Memories

-   **The Importance of a Consistent Data Pipeline**: Many of our challenges stemmed from mismatches in the data pipeline (e.g., `pickle` vs. `json`, default vs. `utf-8` encoding). We learned that ensuring a consistent data format from end to end is critical for success.
-   **The Power of Mode-Agnostic Design**: By refactoring the feature extractors and prediction pipeline to be mode-agnostic, we transformed the project from a niche tool into a powerful, universal utility.
-   **The Value of a Robust Test Suite**: Our test suite was instrumental in identifying and fixing numerous bugs. The final, stable state of the project is a direct result of our commitment to thorough testing.

This project is now in an excellent state, and I am confident it will be a valuable tool for the StepMania community.
