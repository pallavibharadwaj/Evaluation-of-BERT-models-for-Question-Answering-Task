## Evaluation of Language models on Question Answering Tasks

We have considered three model types for comparing on the SQUAD 2.0 dataset.
i. distilbert
ii. roberta
iii. electra-base

**Note**: Replace one of these above model-types in the placeholder in the following command-line options.

### I. Data:

[SQuaD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) should  be present in the data/ folder of the project repository as two JSON files.
    1. Training Data: `./data/train-v2.0.json`
    2. Validation Data: `./data/dev-v2.0.json`

### II. Model Training

1. Running the shell script runs all the models on the SQUAD 2.0 dataset in the sequence.

    `sh ./source/run.sh`

2. Training script generates the best model in the respective model folders `./model/{model-type}/`.


### III. Generating Predictions for Evaluation

1. To run evaluations for a particular model, run the following script that reads the validation data from the `./data/` folder and generates predictions in the `./output` folder under respective model-type folders.

    `python ./source/eval.py {model-type}`

The predictions on dev data is generated on running the above notebook in `./output/{model-type}/output/predictions.json`

### II. Model Evaluation

Model Evaluation can be performed using the script `./check.py`on the predictions and the targets in validation data.

This generates an overall F1 score along with evaluation statistics for questions with amd without answers in the context.

    `python3 evaluate.py ./data/dev-v2.0.json .output/{model-type}/predictions.json`
