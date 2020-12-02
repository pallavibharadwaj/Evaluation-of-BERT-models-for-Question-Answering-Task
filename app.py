from simpletransformers.question_answering import QuestionAnsweringModel
import sys

# context and questions to come from the app
# context: str
# questions: arr of dictionaries {'question': str, 'id': str(<question-num>)}
context = 'Elephants are mammals of the family Elephantidae and the largest existing land animals. Three species are currently recognised: the African bush elephant, the African forest elephant, and the Asian elephant. Elephantidae is the only surviving family of the order Proboscidea; extinct members include the mastodons. The family Elephantidae also contains several now-extinct groups, including the mammoths and straight-tusked elephants. African elephants have larger ears and concave backs, whereas Asian elephants have smaller ears, and convex or level backs. Distinctive features of all elephants include a long proboscis called a trunk, tusks, large ear flaps, massive legs, and tough but sensitive skin. The trunk is used for breathing, bringing food and water to the mouth, and grasping objects. Tusks, which are derived from the incisor teeth, serve both as weapons and as tools for moving objects and digging. The large ear flaps assist in maintaining a constant body temperature as well as in communication. The pillar-like legs carry their great weight.'
questions = [{'question': 'Elephants are mammals of which family?', 'id': '0'}, 
            {'question': 'What is the largest existing land animal?', 'id': '1'},
            {'question': 'How many species are currently recognised?', 'id': '2'},
            {'question': 'What are the existinct members?', 'id': '3'},
            {'question': 'Describe African elephants and Asian elephants', 'id': '4'},  # returns no answer
            {'question': 'Describe African elephants', 'id': '5'},
            {'question': 'Describe Asian elephants', 'id': '6'}]

to_predict = [{'context': context, 'qas': questions}]

# pass the model_type as a parameter
# Ex: python3 app.py electra
model_type = sys.argv[1]

if model_type == "distilbert":
    model_name = "distilbert-base-uncased-distilled-squad"

elif model_type == "roberta":
    model_name = "roberta-base"

elif model_type == "electra-base":
    model_type = "electra"
    model_name = "google/electra-base-discriminator"

# load the best model (hardcode to roberta if needed for the app
model = QuestionAnsweringModel(model_type=model_type, 
                               model_name=f"models/{model_type}/")

print(context)
preds, _ = model.predict(to_predict)

# asnwer for each question within the context
for pred in preds:
    print(questions[int(pred['id'])])
    print(pred['answer'][0])
