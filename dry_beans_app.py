#Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib


#Loading the saved components
model = joblib.load('dry_bean_rforest_model.pkl')
label_encoder = joblib.load('dry_beans_label_encoder.pkl')
scaler = joblib.load('dry_beans_scaler.pkl')


st.title("Dry Beans Type Prediction")
st.write("This model predicts the dry bean type according to the input features below. Please enter your values: ")


Perimeter = st.slider("Perimeter", 200.0, 1400.0)
Extent = st.slider("Extent", 0.60, 0.90)
Solidity = st.slider("Solidity", 0.95, 0.999)
Roundness = st.slider("Roundness", 0.70, 0.999)
Shape_Factor_Two = st.slider("Shape Factor Two", 0.0007, 0.004, format="%f", step=0.0001)
Shape_Factor_Four = st.slider("Shape Factor Four", 0.98, 0.999, step = 0.001)

#Preparing Input Features for Model
features = np.array([[Perimeter, Extent, Solidity, Roundness, Shape_Factor_Two, Shape_Factor_Four]])
scaled_features = scaler.transform(features)

#Prediction
if st.button("Predict Dry Bean Type"):
    prediction_encoded = model.predict(scaled_features)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    st.success(f"Predicted Dry Bean Type: {prediction_label}")