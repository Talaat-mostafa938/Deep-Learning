import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 


st.header('🍓 Image Classification Model')
model = load_model('Image_classify.keras')
data_cat = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']

img_height = 180
img_width = 180

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    st.image(image, caption='Uploaded Image', width=200)

    # Convert image to array
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)  # Create batch axis

    # Prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])

    st.write('Fruit in image is **' + data_cat[np.argmax(score)] + '**')
    st.write('With accuracy of **{:.2f}%**'.format(100 * np.max(score)))