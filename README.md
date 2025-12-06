# StepMania Difficulty Predictor

This project predicts the difficulty of StepMania (.sm) files using a machine learning model.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Predicting Difficulty

### As a Command-Line Tool

To predict the difficulty of a StepMania (.sm) file from the command line:

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the prediction script:**
    ```bash
    python predict_difficulty.py /path/to/your/file.sm
    ```
    The script will output the predicted difficulty for each chart in the file.

You can also specify a custom model file using the `--model-path` argument:
```bash
python predict_difficulty.py /path/to/your/file.sm --model-path /path/to/your/model.p
```

### As a Library

You can also use this project as a library in your own Python code.

1.  **Install the package:**
    ```bash
    pip install .
    ```

2.  **Use the `DifficultyPredictor` class:**
    ```python
    from src.models.prediction_pipeline import DifficultyPredictor

    # Initialize the predictor with the path to the model
    predictor = DifficultyPredictor("models/random_forest_regressor.p")

    # Predict the difficulty of a .sm file
    predictions = predictor.predict("path/to/your/file.sm")

    for p in predictions:
        print(f"Difficulty: {p['difficulty']}, Meter: {p['meter']}, Predicted Difficulty: {p['predicted_difficulty']:.2f}")
    ```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
