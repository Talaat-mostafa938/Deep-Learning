import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import cv2 as cv
import numpy as np 
from PIL import Image



st.header('Brain Tumor Classification 🧠')
model = load_model('brain_tumor_detection_model.h5')
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

img_height = 224
img_width = 224

uploaded_file = st.file_uploader("Upload an Image" , type=['jpg' , 'jpeg' , 'png'])


if uploaded_file is not None:
    image_path = tf.keras.utils.load_img(uploaded_file , target_size = (img_height, img_width))
    st.image(image_path , caption = 'Uploaded Image' , width = 200)
    
    # Convert image to array
    image_pil = Image.open(uploaded_file).convert('RGB')
    image_array  = tf.keras.utils.img_to_array(image_pil)
    img_gray = cv.cvtColor(image_array , cv.COLOR_RGB2GRAY)
    img_resized = cv.resize(img_gray , (img_height, img_width))
    img_norm = img_resized / 255
    img_input = img_norm.reshape(1 , img_height, img_width, 1)
    
    predict = model.predict(img_input)
    score = np.argmax(predict[0])
    
    st.write('Tumor in image is **' + classes[score] + '**')
    st.write('With accuracy of **{:.2f}%**'.format(100 * np.max(predict[0])))
