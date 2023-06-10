# Import all of the dependencies
import streamlit as st
import os 

import tensorflow as tf 
from modelutils import decode_batch_predictions, preprocess_binary_img_png, prediction_model

# Set the layout to the streamlit app as wide 
# st.set_page_config(layout='wide')
st.title('Captcha reader')
image = st.file_uploader(label="Upload a captcha image", type="png")
if image:
    st.image(image)
    image = image.getvalue()
    image = preprocess_binary_img_png(image)
    pred = prediction_model.predict(image)
    pred = decode_batch_predictions(pred)
    st.write(pred[0])
