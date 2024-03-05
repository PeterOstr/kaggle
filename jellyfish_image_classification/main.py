from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input, decode_predictions  # Add this import
import pandas as pd
import numpy as np
from keras.models import load_model
from pydantic import BaseModel
import tensorflow as tf
from PIL import Image
from io import BytesIO
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

model = load_model('model_new.h5')

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


# Load the pre-trained ResNet50 model from a local file
model_50 = load_model('resnet50_model.h5')  # Specify the path to the local model file

@app.post("/classify_image/")
async def classify_image(image_file: UploadFile = File(...)):
    # Read image file
    contents = await image_file.read()

    # Convert image bytes to PIL Image object
    img = Image.open(BytesIO(contents))

    # Resize image to match ResNet50 input size
    image_resized = img.resize((224, 224))
    image_tensor = tf.keras.preprocessing.image.img_to_array(image_resized)

    image_tensor_preprocessed = preprocess_input(image_tensor)

    # Compile the model manually
    model_50.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # Add batch dimension and preprocess input
    #preds = model_50.predict(image_tensor_preprocessed)
    preds = model_50.predict(np.expand_dims(image_tensor_preprocessed, axis=0))



    # Decode predictions
    decoded_preds = decode_predictions(preds, top=1)[0][0][1]  # Top 3 predictions

    # Format predictions
    #predictions = [{"class_name": class_name, "probability": float(prob)} for (_, class_name, prob) in decoded_preds]

    #return {"predictions": predictions}
    return decoded_preds