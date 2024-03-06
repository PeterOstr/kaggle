import pandas as pd
import numpy as np
import streamlit as st
import requests

# Header and text
st.title("Image classification using the Jellyfish Image Dataset as an example")
st.write("""This dashboard will present the info about the COVID-19 dataset, taken from 
Kaggle platform (https://www.kaggle.com/datasets/anshtanwar/jellyfish-types/data)).""")

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
        url1 = "https://jellyapp-unk2qlgpqa-lm.a.run.app/classify_image/"
        url2 = "https://jellyapp-unk2qlgpqa-lm.a.run.app/model/predict"



        files1 = {'image_file': uploaded_file}
        files2 = {'image_file': uploaded_file}

        response1 = requests.post(url1, files=files1)
        response2 = requests.post(url2, files=files2)

        # Check the response code and display the results
        if response1.status_code == 200:
            result = response1.json()
            st.write('Predictions:')
            # for prediction in result['predictions']:
            #     st.write(f"Class: {prediction['class_name']}, Probability: {prediction['probability']}")
            st.write(result)
        else:
            st.error(f"Error sending the image: {response1.status_code}")

        if response2.status_code == 200:
            result2 = response2.json()
            st.write('Class:')
            st.write(result2)
        else:
            st.error(f"Error sending the image: {response2.status_code}")


if st.button('Send2'):
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        files = {'image_file': uploaded_file}

    # Send the image to the backend
        # Define the URLs for the two endpoints
        url1 = "https://jellyapp-unk2qlgpqa-lm.a.run.app/classify_image/"
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