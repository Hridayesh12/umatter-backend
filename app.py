from flask import Flask, jsonify, request
import numpy as np
from model import *
from flask_cors import CORS

app=Flask(__name__)
CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})

@app.route('/')
def hello():
    return "Hello"
@app.route('/postSurveyData', methods=['POST'])
def surveyDatas():
    surveyDa = request.get_json()
    list=[]
    for x in surveyDa:
        list.append(surveyDa[x])
    print(list)
    predicted_value=clf.predict([list])
    predicted_value=predicted_value.tolist()
    print("Prediction value: ",predicted_value)
    if not surveyDa:
    #     return predicted_value, 400
    # return predicted_value, 200
        return jsonify({'msg': predicted_value}), 400
    return jsonify({'msg': predicted_value}), 200
if __name__=='__main__':
    app.run(debug=True)