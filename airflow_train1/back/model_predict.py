from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
from typing import List
from pydantic import BaseModel
import catboost
from catboost import CatBoostRegressor
from datetime import datetime
from io import StringIO



import joblib

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

class Date_for_prediction_data(BaseModel):
    #index: List
    date: List


model_xgb_t = joblib.load('my_xgb_model.joblib')
model_catb_t = joblib.load('my_catb_model.joblib')

# using bin-saved model as alternative
catboost_model = CatBoostRegressor()
catboost_model.load_model('catboost_model.bin')


@app.post("/model/predict_xgb_t")
async def predict_xgb_t(data: Date_for_prediction_data):
    print(data)
    data_str = StringIO(data.model_dump_json())
    print(data_str)

    data_df = pd.read_json(data_str, orient='index')
    #print(type(data_df))

    return data_df

    # data_df.columns = ['Local time in Moscow']
    # print(data_df)

    #data_df['Local time in Moscow'] = pd.to_datetime(data_df['Local time in Moscow'], format='%Y-%m-%d %H:%M:%S')
    # print(type(data))
    #
    # #data_df = pd.DataFrame.from_dict(pd.read_json(data)['date'], orient='index', columns=['date'])
    #
    # # Преобразуйте новые данные в DataFrame и укажите правильное имя столбца
    # print(data_df)
    #
    # # Извлеките признаки из новых данных
    # data_features = pd.DataFrame()
    # data_features['year'] = data_df['Local time in Moscow'].dt.year
    # data_features['month'] = data_df['Local time in Moscow'].dt.month
    # data_features['day'] = data_df['Local time in Moscow'].dt.day
    # data_features['hour'] = data_df['Local time in Moscow'].dt.hour
    #
    #
    #
    # #model_prediction = round(model_xgb_t.predict(data_features)[0],2)
    # #model_prediction = model_prediction.to_json()
    # return data
    #return {"prediction": float(model_prediction)}  # Преобразуйте прогноз в тип float перед возвратом


@app.post("/model/predict_catb_t")
async def predict_catb_t(data: Date_for_prediction_data):
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



    model_prediction = round(model_catb_t.predict(data_features)[0],2)
    #model_prediction = model_prediction.to_json()
    #return model_prediction
    return {"prediction": float(model_prediction)}  # Преобразуйте прогноз в тип float перед возвратом


@app.post("/model/predict_catb_t_alt")
async def predict_catb_t(data: Date_for_prediction_data):
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



    model_prediction = round(catboost_model.predict(data_features)[0],2)

    return {"prediction": float(model_prediction)}  # Преобразуйте прогноз в тип float перед возвратом