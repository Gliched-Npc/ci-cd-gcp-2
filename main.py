"""Prediction using fast api 
"""

import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

model=joblib.load('model/model.joblib')


# This class likely represents a model for predicting iris species based on certain features.
class IrisModel(BaseModel):
    """
    Getting model predction features as list 
    """
    features:list[float]

LABELS=['Setosa','Virginica','Versicolor']

@app.get('/')
def root():
    """
    The function is for health check up 
    """
    return {'Health':'Ok'}

@app.post('/predict')
def prediciton(data:IrisModel):
    """
    The function takes input data, makes a prediction using a pre-trained model, and returns the
    predicted class index and class name
    """
    x=np.array([data.features])
    pred=int(model.predict(x)[0])
    return {
        'class-index': pred,
        'class-name':LABELS[pred]
    }
