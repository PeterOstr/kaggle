import pandas as pd
import numpy as np
import streamlit as st
import requests

# Header and text
st.title("Image classification using the Jellyfish Image Dataset as an example")
st.write("""This is training dashboard with the info about the Jellyfish, taken from 
Kaggle platform (https://www.kaggle.com/datasets/anshtanwar/jellyfish-types/data)).""")

with st.expander("See notes"):
    st.write("""
        * We predicting 6 types of Jellyfish:
            - Moon jellyfish (Aurelia aurita): Common jellyfish with four horseshoe-shaped gonads visible through the 
            top of its translucent bell. It feeds by collecting medusae, plankton, and mollusks with its tentacles.
            - Barrel jellyfish (Rhizostoma pulmo): Largest jellyfish found in British waters, with a bell that can grow
             up to 90 cm in diameter. It feeds on plankton and small fish by catching them in its tentacles.
            - Blue jellyfish (Cyanea lamarckii): Large jellyfish that can grow up to 30 cm in diameter. It feeds on 
            plankton and small fish by catching them in its tentacles.
            - Compass jellyfish (Chrysaora hysoscella): Named after the brown markings on its bell that resemble a 
            compass rose. It feeds on plankton and small fish by catching them in its tentacles.
            - Lion’s mane jellyfish (Cyanea capillata): Largest jellyfish in the world, with a bell 
            that can grow up to 2 meters in diameter and tentacles that can reach up to 30 meters in length. It feeds on plankton and small fish by catching them in its tentacles.
            - Mauve stinger (Pelagia noctiluca): Small jellyfish with long tentacles 
            and warty structures on its bell full of stinging cells. It feeds on other small jellyfish and oceanic sea squirts.
        
        """)

# Button to upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

# Button to send image to the backend
if st.button('Send'):
    if uploaded_file is not None:
        # Send the image to the backend
        # Define the URLs for the two endpoints
        url = "https://jellyapi-unk2qlgpqa-lm.a.run.app/model/predict"



        files = {'image_file': uploaded_file}

        response = requests.post(url, files=files)

        # Check the response code and display the results
        if response.status_code == 200:
            result = response.json()
            st.write('Predictions:')
            # for prediction in result['predictions']:
            #     st.write(f"Class: {prediction['class_name']}, Probability: {prediction['probability']}")
            st.write(result[0])
        else:
            st.error(f"Error sending the image: {response.status_code}")



if st.button('Send2'):
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        files = {'image_file': uploaded_file}

    # Send the image to the backend
        # Define the URLs for the two endpoints
        url1 = "https://jellyapi-unk2qlgpqa-lm.a.run.app/classify_image/"
        url2 = "https://jellyapp-unk2qlgpqa-lm.a.run.app/model/predict"
        response2 = requests.post(url2, files=files)

        if response2.status_code == 200:
            result2 = response2.json()
            st.write('Class:')
            st.write(result2)
        else:
            st.error(f"Error sending the image: {response2.status_code}")

st.sidebar.info("Feel free to contact me\n"'\n'
                "[My GitHub](https://github.com/PeterOstr)\n"'\n'
                "[My Linkedin](https://www.linkedin.com/in/ostrikpeter/)\n"'\n'
                "[Or just text me in Telegram](https://t.me/Politejohn)\n"
                ".")

st.write(""" ### About this app here

Initializing the FastAPI Application: A web application is created using FastAPI, which handles HTTP requests.

Image Processing: Users can upload images through the web interface of the application.

Predicting Jellyfish Type in the Image: Upon uploading an image, the application utilizes a neural network to 
analyze the image and predict the type of jellyfish present in it.

Image Classification using ResNet50: Additionally, the application can classify the image using a pre-trained 
ResNet50 model to determine its content.

Application Response: After processing the image, the application returns the result to the user, such as the
 predicted jellyfish type or image classification.

Error Handling: The application ensures handling of potential errors that may occur during image upload and processing.
""")