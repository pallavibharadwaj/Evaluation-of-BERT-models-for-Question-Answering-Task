from simpletransformers.question_answering import QuestionAnsweringModel
import sys
import os
import json
import transformers
import torch

# load the dev data into dictionary
with open('../data/dev-v2.0.json', 'r') as f:
    dev_data = json.load(f)    
dev_data = [item for topic in dev_data['data'] for item in topic['paragraphs'] ]

print("===Loaded Dev data===")
print("Dev data size = ", len(dev_data))

model_type = sys.argv[1]

if model_type == "distilbert":
    model_name = "distilbert-base-uncased-distilled-squad"

elif model_type == "roberta":
    model_name = "roberta-base"

elif model_type == "electra-base":
    model_type = "electra"
    model_name = "google/electra-base-discriminator"

elif model_type == "electra-small":
    model_type = "electra"
    model_name = "google/electra-small-discriminator"

train_args = {
    "reprocess_input_data": False,
    "overwrite_output_dir": True,
    "use_cached_eval_features": True,
    "output_dir": f"../models/{model_type}",
    "best_model_dir": f"../models/{model_type}/best_model",
    "max_seq_length": 128,
    "num_train_epochs": 2,
    "wandb_project": "QuestionAnswering Model Comparison",
    "wandb_kwargs": {"name": model_name},
    "train_batch_size": 8,

    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
}

# load the trained model
model = QuestionAnsweringModel(model_type=model_type, 
                               model_name=f"../models/{model_type}", 
                               args=train_args, 
                               use_cuda=True)
print("===Loaded fine-tuned model for predictions===")

# make predictions on dev data
preds, _ = model.predict(dev_data)
print("===Predicted on dev data successfully===")

predictions = {pred['id']: pred['answer'][0] for pred in preds}
print("prediction size = ", len(predictions))

# write predictions to file
os.makedirs(f"../output/{model_type}", exist_ok=True)
with open(f"../output/{model_type}/predictions.json", 'w') as f:
    json.dump(predictions, f)
print("Wrote predictions to output directory")
print("run \"python check.py data/dev-v2.0.json output/{model_type}/predictions.json\" in project directory for validation F1 score.")
