# StepMania Difficulty Predictor

This project predicts the difficulty of StepMania (.sm) files using a machine learning model.

## Installation

You can install the package using pip:
```bash
pip install .
```

## Usage

### As a Command-Line Tool

To predict the difficulty of a StepMania (.sm) file from the command line:

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

```python
from stepmania_difficulty_predictor.models.prediction_pipeline import DifficultyPredictor
import simfile

# Initialize the predictor. By default, it will load the packaged model.
# You can also provide a path to a custom model:
# predictor = DifficultyPredictor("path/to/your/model.p")
predictor = DifficultyPredictor()

# Predict the difficulty of a .sm file from a path
predictions_from_path = predictor.predict("path/to/your/file.sm")

for p in predictions_from_path:
    print(f"Difficulty: {p['difficulty']}, Meter: {p['meter']}, Predicted Difficulty: {p['predicted_difficulty']:.2f}")

# You can also predict the difficulty of a simfile object
with open("path/to/your/file.sm", "r") as f:
    sm = simfile.open(f)
    predictions_from_object = predictor.predict(sm)

    for p in predictions_from_object:
        print(f"Difficulty: {p['difficulty']}, Meter: {p['meter']}, Predicted Difficulty: {p['predicted_difficulty']:.2f}")
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
