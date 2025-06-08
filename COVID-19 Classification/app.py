import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import cv2 as cv
import numpy as np 
from PIL import Image



st.header('COVID-19 X-ray Detection 🫁')
model = load_model('COVID-19 Xray Detection.h5')
data_cat = ['NORMAL', 'PNEUMONIA']

img_height = 224
img_width = 224

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_file , caption='Uploaded Image', width=200)

    # Convert image to array
    
    image_np = tf.keras.utils.img_to_array(image_pil)
    img_gray = cv.cvtColor(image_np , cv.COLOR_RGB2GRAY)
    img_resized = cv.resize(img_gray , (img_height , img_width))
    img_norm = img_resized / 255
    img_input = img_norm.reshape(1,img_height , img_width , 1)
    

    # Prediction
    predict = model.predict(img_input)[0][0]
    if predict >= .5:
        pred = "PNEUMONIA"
        
        st.write('Case detected is" **' + pred +'**')
        st.write('With accuracy of **{:.2f}%**'.format(100 * predict))
    else:
        pred = "NORMAL"
        st.write('Case detected is" **' + pred +'**')
        st.write('With accuracy of **{:.2f}%**'.format(100 * (1 - predict)))