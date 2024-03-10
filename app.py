import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
from flask_cors import CORS
import numpy as np
import pandas as pd

app=Flask(__name__)
CORS(app)
## Load the model
classifier_model=pickle.load(open('classifier.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(data.values())
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=classifier_model.predict(new_data)
    response_to_be_send = 'Diabetic' if output[0]==1 else 'Non-Diabetic'
    # Convert output to a serializable data type
    return jsonify(response_to_be_send)


if __name__=="__main__":
    app.run(debug=True)