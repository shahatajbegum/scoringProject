import mlflow 
import uvicorn
import json
import pandas as pd 
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile
import joblib
import lightgbm
import imblearn

description = """
"""

tags_metadata = [
]

app = FastAPI(
)
class PredictionFeatures(BaseModel):
    Id: Union[int, float]

@app.post("/predict", tags=["Machine-Learning"])
async def predict(predictionFeatures: PredictionFeatures):
    """
    Prediction for one observation. Endpoint will return a dictionnary like this:
    ```
    {'prediction': PREDICTION_VALUE[0,1]}
    ```
    You need to give this endpoint all columns values as dictionnary, or form data.
    """
    # Read data 
    
    df = pd.read_csv("data.csv")
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df.set_index('SK_ID_CURR' ,inplace=True)
    print("*"*50)
    print("test amine")
    print(str(dict(predictionFeatures)['Id']))
    print(type(dict(predictionFeatures)['Id']))
    df = df[df.index==dict(predictionFeatures)['Id']]


    # Load model as a PyFuncModel.
    loaded_model = joblib.load("lgbm.joblib")
    prediction = round(loaded_model.predict_proba(df)[0][1],3)
    # Format response
    response = {"prediction": prediction}
    
    return response



if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, debug=True, reload=True)