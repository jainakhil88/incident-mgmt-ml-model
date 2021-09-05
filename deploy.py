# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:36:32 2021

@author: Akhil Jain
"""

from flask import render_template, request, Response
from flask import Flask
from incident_mgmt_model import Model
import pandas as pd
import io

app = Flask(__name__)  

model = Model()
 
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")
 
@app.route('/predict', methods = ['POST'])  
def predict():  
    if request.method == 'POST':          
        rawDataFrame = pd.read_csv(request.files.get('file'))
        
        #passing dataframe to model
        score=model.predict(rawDataFrame)    
                
        return Response(response=score)
  
if __name__ == '__main__':  
    app.run(debug = True , use_reloader=False)  
    #