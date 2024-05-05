from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
from typing import List
from pydantic import BaseModel
import joblib

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

class Date_for_prediction_data(BaseModel):
    index: List
    date: List


model_xgb_t = joblib.load('my_xgb_model.joblib')


@app.post("/model/predict_xgb_t")
async def predict_xgb_t(data: Date_for_prediction_data):
    data = pd.read_json(data.model_dump_json(), orient='list')
    data = data.set_index('index')

    # Преобразуйте новые данные в DataFrame и укажите правильное имя столбца
    data_df = pd.DataFrame(data, columns=['Local time in Moscow'])
    data_df['Local time in Moscow'] = pd.to_datetime(data_df['Local time in Moscow'], format='%Y-%m-%d %H:%M:%S')

    # Извлеките признаки из новых данных
    data_features = pd.DataFrame()
    data_features['year'] = data_df['Local time in Moscow'].dt.year
    data_features['month'] = data_df['Local time in Moscow'].dt.month
    data_features['day'] = data_df['Local time in Moscow'].dt.day
    data_features['hour'] = data_df['Local time in Moscow'].dt.hour



    model_prediction = round(model_xgb_t.predict(data_features)[0],2)
    #model_prediction = model_prediction.to_json()
    #return model_prediction
    return {"prediction": float(model_prediction)}  # Преобразуйте прогноз в тип float перед возвратом
