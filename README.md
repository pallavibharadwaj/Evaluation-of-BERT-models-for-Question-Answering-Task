
## Evaluation of Language models on Question Answering Tasks

  ### Prerequisites

1. Create a python3 virtual environment inside the project folder and install the required dependencies.

    `python3 -m qa_env venv` \
    `source qa_env/bin/activate` \
    `pip3 install -r requirements.txt`

2. Download the `data` and `models` folders as required into the top-level of the project repository from `https://vault.sfu.ca/index.php/s/JhnegF4Qs7ZTH4R` .

3. We have considered three model types for comparing on the SQUAD 2.0 dataset. Replace one of the following model-types in place of the placeholder in the following command-line options.

		1. distilbert
		2. roberta
		3. electra-base

There are two ways of using this repository depending on the needs of the user:

I. [Evaluation of BERT Models](#Evaluation-of-BERT-models) \
II. [Web Application for Question Answering](#Web-Application-for-Question-Answering)
 
###  I.  Evaluation of BERT models

In order to do this, you will need the files to be in the following directory structure. The `models` folder is generated/re-written on running the `./train/run.sh` shell script.

```
├── data 
      └── train-v2.0.json
      └── dev-v2.0.json
├── models 
      └── distilbert
      └── electra
      └── roberta
├── train
      └── run.sh
      └── train.py
      └── eval.py
├── output
      └── distilbert
            └── predictions.json
      └── electra
            └── predictions.json
      └── roberta
            └── predictions.json
├── check.py
```
#### Data:

We have used the [SQuaD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset to train our BERT models. The data consists of two JSON files one each for train and evaluation. These files should be present in the `./data` folder of the project repository as described above.

1. Training Data: `./data/train-v2.0.json`
2. Validation Data: `./data/dev-v2.0.json`

#### Model Training

1. Running the shell script runs all the models on the SQUAD 2.0 dataset in a sequence. \

    `sh ./source/run.sh`
2. Training script generates the best model in the respective model folders `./model/{model-type}/`.

#### Generating Predictions on Dev Data

1. To run evaluations for a particular model, run the following script that reads the validation data  `./data/dev-v2.0.json` folder and generates predictions in the `./output` folder under respective model-type folders. \

    `python ./source/eval.py {model-type}`
2. The predictions on dev data is generated on running the above notebook in `./output/{model-type}/output/predictions.json`

#### Model Evaluation

1. Model Evaluation can be performed using the script `./check.py`on the predictions and the targets in validation data.
2. This generates an overall F1 score along with evaluation statistics for questions with amd without answers in the context. \

    `python3 evaluate.py ./data/dev-v2.0.json .output/{model-type}/predictions.json`

#### Result

|                   |**DistilBERT** | **ELECTRA** | **RoBERTa** |
| ----------------- | ------------- | ----------- | ----------- |
| **HasAns_exact**  | 47.048        | 57.557      | 58.87       |
| **HasAns_f1**     | 51.75         | 63.94       | 63.65       |
| **NoAns_exact**   | 69.45         | 70.06       | 86.53       |
| **NoAns_f1**      | 69.45         | 70.06       | 86.53       |
| **exact**         | 58.27         | 63.82       | 72.72       |
| **f1**            | 60.61         | 67.00       | 75.10       |

* RoBERTa is a clear winner - -   Pre-trained on a much bigger training data (161G) compared to BERT (16G), on longer sequences, usinf dynamic masking as opposed to static masking techniques used in BERT.

![](./assets/loss.png?raw=true "Relative Loss")

* The performance of DistilBERT is fair considering the simplicity of the model, shorter runtime using much lesser computational resources.

Relative Runtimes                                                | Relative GPU Utilization
:---------------------------------------------------------------:|:-----------------------------------------------------------------------:
![](./assets/relative_runtime.png?raw=true "Relative Runtimes")  |  ![](./assets/relative_gpu_util.png?raw=true "Relative GPU Utilization")


 # II. Web Application for Question Answering

In order to run the web application, make sure that our best performing model, roberta, is present in the `./models` folder at the top-level of the project repository.

The folder structure should look like the following:
```
└── models
     └── roberta
├── web_application 
     └── app.py
     └── app.test.py
     └── static
           └── css
           └── js
     └── templates
           └── base.html
           └── home.html
```

#### Running the Web Application

1. Enter the `web_application` folder and just run the python script `app.py`.

    `cd ./web_application`  \
    `python3 app.py`

2. Run the application you see on the terminal on a  browser. (Ex: http://127.0.0.1:5000/).

3. You will see an application as following that takes a `context` paragraph with a `question` and returns the `answer` using our best performing model `roberta`.

![](./assets/web_app.png?raw=true "Demo Web Application")

### Project Contributers

1. Pallavi Bharadwaj - pallavib@sfu.ca
2. Najeeb Qazi - nqazi@sfu.ca

### References

[1] Yiling Chen and Rena Sha. Question Answering on Natural Questions. 2019. \
[2] [Kevin Clark](https://openreview.net/profile?email=kevclark%40cs.stanford.edu), [Minh-Thang Luong](https://openreview.net/profile?email=thangluong%40google.com), [Quoc V. Le](https://openreview.net/profile?email=qvl%40google.com), [Christopher D. Manning](https://openreview.net/profile?email=manning%40cs.stanford.edu) - “ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators”. In: International Conference on Learning Representations. 2020. URL: [https://openreview.net/forum?id=r1xMH1BtvB](https://openreview.net/forum?id=r1xMH1BtvB). \
[3]  [Yinhan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Myle Ott](https://arxiv.org/search/cs?searchtype=author&query=Ott%2C+M), [Naman Goyal](https://arxiv.org/search/cs?searchtype=author&query=Goyal%2C+N), [Jingfei Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+J), [Mandar Joshi](https://arxiv.org/search/cs?searchtype=author&query=Joshi%2C+M), [Danqi Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+D), [Omer Levy](https://arxiv.org/search/cs?searchtype=author&query=Levy%2C+O), [Mike Lewis](https://arxiv.org/search/cs?searchtype=author&query=Lewis%2C+M), [Luke Zettlemoyer](https://arxiv.org/search/cs?searchtype=author&query=Zettlemoyer%2C+L), [Veselin Stoyanov](https://arxiv.org/search/cs?searchtype=author&query=Stoyanov%2C+V) - A Robustly Optimized BERT Pretraining Approach. 2019. arXiv: [1907.11692 [cs.CL]](https://arxiv.org/abs/1907.11692) \
[4] Pranav Rajpurkar, Robin Jia, and Percy Liang - Know What You Don’t Know: Unanswerable Questions for SQuAD. 2018. arXiv: [1806.03822 [cs.CL].](https://arxiv.org/abs/1806.03822) \
[5] [Victor Sanh](https://arxiv.org/search/cs?searchtype=author&query=Sanh%2C+V), [Lysandre Debut](https://arxiv.org/search/cs?searchtype=author&query=Debut%2C+L), [Julien Chaumond](https://arxiv.org/search/cs?searchtype=author&query=Chaumond%2C+J), [Thomas Wolf](https://arxiv.org/search/cs?searchtype=author&query=Wolf%2C+T) - DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. 2020. arXiv:  [1910.01108 [cs.CL]](https://arxiv.org/abs/1910.01108).
