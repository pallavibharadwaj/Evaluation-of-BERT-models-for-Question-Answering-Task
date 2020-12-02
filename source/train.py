from simpletransformers.question_answering import QuestionAnsweringModel
import sys
import os
import json
import transformers
import torch
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
torch.cuda.is_available()

# load train data into a dictionary
with open('../data/train-v2.0.json', 'r') as f:
    train_data = json.load(f)
train_data = [item for topic in train_data['data'] for item in topic['paragraphs'] ]

print("===Loaded Training data===")
print("Train data size = ", len(train_data))

model_type = sys.argv[1]

if model_type == "distilbert":
    model_name = "distilbert-base-uncased-distilled-squad"

elif model_type == "roberta":
    model_name = "deepset/roberta-base-squad2"

elif model_type == "electra-base":
    model_type = "electra"
    model_name = "deepset/electra-base-squad2"

train_args = {
    "reprocess_input_data": False,
    "overwrite_output_dir": True,
    "use_cached_eval_features": True,
    "output_dir": f"../models/{model_type}",
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

model = QuestionAnsweringModel(model_type=model_type, 
                               model_name=model_name,
                               args=train_args, 
                               use_cuda=True,
                               cuda_device=0)
print("===Loaded Pre-trained DistilBert Model===")


# fine-tuning 
os.makedirs(f"../models/{model_type}", exist_ok=True)
model.train_model(train_data=train_data, args=train_args)
print("===Trained the DistilBert Model and saved fine-tuned model===")
