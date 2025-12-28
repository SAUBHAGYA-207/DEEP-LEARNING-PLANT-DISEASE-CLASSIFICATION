import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pickle
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from huggingface_hub import hf_hub_download
import time

def prepare_single_image(uploaded_file, img_size=224):
  
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return img
plant_encoder=pickle.load(open("plant_encoder.pkl","rb"))
disease_encoder=pickle.load(open("disease_encoder.pkl","rb"))
plant_classes=list(plant_encoder.classes_)
disease_classes=list(disease_encoder.classes_)
repo_id = "100xFORTUNE/plant_disease_classification"



st.title("LeafCheck")
st.subheader("Plant and Disease Classfication")
with st.spinner("Loading model... Please wait", show_time=True):
    model_path = hf_hub_download(
    repo_id=repo_id,
    filename="plant_model.keras" )
    model = tf.keras.models.load_model(model_path)
st.success("Model loaded successfully!")



uploaded_file = st.file_uploader("Select Your Sample",type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Sample", use_container_width=True)
    progress_text = "Processing image..."
    my_bar = st.progress(0, text=progress_text)
    img=prepare_single_image(uploaded_file)
    my_bar.progress(30, text="Image preprocessed")

    pred = model.predict(img)
    my_bar.progress(70, text="Model prediction done")
    st.success("Prediction completed")

    plant_pred=pred['plant_output']
    disease_pred=pred['disease_output']

    plant_id = np.argmax(plant_pred)
    disease_id = np.argmax(disease_pred)

    plant_name = plant_classes[plant_id]
    disease_name = disease_classes[disease_id]

    plant_conf = np.max(plant_pred)
    disease_conf = np.max(disease_pred)
    my_bar.progress(100, text="Finalizing results")
    time.sleep(0.5)
    my_bar.empty()
    st.markdown(f"Our Model detected the sample to be {plant_name} with {plant_conf:.2%} confidence")
    st.markdown(f"Our Model detected the sample to be {disease_name} with {disease_conf:.2%} confidence")






