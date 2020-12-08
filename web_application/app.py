from flask import Flask, render_template, make_response, request, redirect, url_for, jsonify
from flask_restful import Api, Resource, reqparse
from simpletransformers.question_answering import QuestionAnsweringModel
import sys


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return 'about page'

@app.route('/data', methods = ['POST'])
def get_data():
	if request.method == 'POST':
		if(request.get_json() is None):
			data = request.form
		else:
			data = request.get_json()
		context = data['context']
		question = data['question']

		to_predict = [{'context': context, 'qas': [{'question':question,'id':'0'}]}]

		model_type = "roberta"
		model = QuestionAnsweringModel(model_type=model_type, 
                               model_name=f"models/{model_type}/", use_cuda = False)

		preds, _ = model.predict(to_predict)

		print(preds[0]['answer'][0])
		if(preds[0]['answer'][0] == ""):
			result = "No answer found"
		else:
			result = preds[0]['answer'][0]

		return jsonify({'output':result})


@app.route('/<page_name>')
def other_page(page_name):
    response = make_response('The page named %s does not exist.' \
                            % page_name, 404)
    return response


if __name__ == '__main__':
    app.run(debug=True)
