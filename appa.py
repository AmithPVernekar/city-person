from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
import os

import pandas as pd
from downcast import reduce
from tqdm import tqdm
import pickle


app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict1", methods=['POST'])
def predict1():
    if request.method == 'POST':
        text = str(request.form['text'])
        
        pickle_in = open("bow.pickle","rb")
        bow = pickle.load(pickle_in)
        transformed_text=bow.transform([text])
        
        
        pickle_in = open("model.pickle","rb")
        clf = pickle.load(pickle_in)
        output=clf.predict(transformed_text.toarray())
        if output[0]==0:
            output="City"
        else:
            output="Person"
        output_text = text+" is a "+output+" name"
    return render_template('index.html',predicted_class=output_text)
if __name__=="__main__":  
    app.run(host='0.0.0.0',port='8080',debug=True)
    #app.run(debug=True)

