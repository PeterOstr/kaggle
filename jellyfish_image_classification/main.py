from fastapi import FastAPI, UploadFile, File
from typing import List
import pandas as pd
import numpy as np
from keras.models import load_model
from pydantic import BaseModel
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

class ImageData(BaseModel):
    image_file: UploadFile

jelly_type_df = pd.DataFrame({
    0: ['Moon_jellyfish'],
    1: ['barrel_jellyfish'],
    2: ['blue_jellyfish'],
    3: ['compass_jellyfish'],
    4: ['lions_mane_jellyfish'],
    5: ['mauve_stinger_jellyfish']
}).T

model = load_model('C:/Users/Peter/DataspellProjects/kaggle/jellyfish_image_classification/model_new.h5')

@app.post("/model/predict")
async def predict(image_file: UploadFile = File(...)):
    try:
        contents = await image_file.read()
        image = Image.open(io.BytesIO(contents))
        image_resized = image.resize((224, 224))
        image_tensor = tf.keras.preprocessing.image.img_to_array(image_resized)
        result = model.predict(np.expand_dims(image_tensor, axis=0))
        max_index = np.argmax(result)
        predicted_jellyfish_type = jelly_type_df.iloc[max_index, 0]
        return predicted_jellyfish_type
    except Exception as e:
        return {"error": str(e)}