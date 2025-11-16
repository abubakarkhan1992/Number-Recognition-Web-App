#Assignment 4

import streamlit as st
import pickle
from PIL import Image
import numpy as np

st.set_page_config("Number Recognition App", layout="centered")
st.title("Number Recognition App")

model = pickle.load(open("number_recognition.pkl", "rb"))

st.subheader("Upload image of a digit between 0 and 9")
iamge = st.file_uploader(label = "image", type=["jpg", "jpeg", "png"])

def predict_number(image):
    original_image = Image.open(image)
    grayscale_image = original_image.convert("L")

    grayscale_array = np.array(grayscale_image)/255
    grayscale_array = grayscale_array.reshape(1,784)

    model.predict(grayscale_array)
    return model.predict(grayscale_array).argmax()

if iamge is not None:
    st.image(iamge)
    prediction = predict_number(iamge)
    st.success(f"The predicted number is: {prediction}")



