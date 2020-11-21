## Evaluation of Language models on Question Answering Tasks

### I. Data:

[SQuaD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) should  be present in the data/ folder of the project repository as two JSON files.
    1. data/train-v2.0.json
    2. data/dev-v2.0.json

### II. Model Training

1. Distilbert Model is available in the jupyter-notebook `source/distilbert/distilbert.ipynb`

2. Running this file generates checkpoints and model files that can be used later for predictions on dev and test data in `jupyter-notebook source/distilbert/checkpoints`

### II. Model Evaluation

1. The predictions on dev data is generated on running the above notebook in `source/distilbert/output/predictions.json`

2. Model Evaluation can be performed using the script `evaluate.py`on the predictions and the dev data. This generates the F1 score and other evaluation statistics.

    `python3 evaluate.py data/dev-v2.0.json source/distilbert/output/predictions.json`
