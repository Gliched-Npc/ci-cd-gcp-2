import os 
import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X,y=load_iris(return_X_y=True)

model=RandomForestClassifier()
model.fit(X,y)

os.makedirs('model',exist_ok=True)
joblib.dump(model,'model/model.joblib')

print('model is created')