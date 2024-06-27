import os
import pickle
import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="Failure Classifier",
    page_icon="images/icon.png",
)

# Page title
st.title('Machine - Failure Prediction')
st.image('images/maintenance.jpg')
st.write("\n\n")

st.markdown(
    """
   This application facilitates the rapid classification of machine failures using sensor data, expediting the troubleshooting process.
    """
)

# Print current working directory
st.write(f"Current working directory: {os.getcwd()}")

# Check if the model file exists
model_path = 'Model/model (3).pkl'
if not os.path.exists(model_path):
    st.error(f"Model file not found at path: {model_path}")

# Load the model
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit interface to input data
col1, col2 = st.columns(2)

with col1:
    air = st.number_input(label='Air Temperature')
    process = st.number_input(label='Process Temperature')
    rpm = st.number_input(label='Rotational Speed')

with col2:
    torque = st.number_input(label='Torque')
    tool_wear = st.number_input(label='Tool Wear')
    type = st.selectbox(label='Type', options=['Low', 'Medium', 'High'])

# Function to predict the input
def prediction(air, process, rpm, torque, tool_wear, type):
    # Create a df with input data
    df_input = pd.DataFrame({
        'Air_temperature': [air],
        'Process_temperature': [process],
        'Rotational_speed': [rpm],
        'Torque': [torque],
        'Tool_wear': [tool_wear],
        'Type': [type]
    })

    prediction = model.predict(df_input)
    return prediction

# Button to predict
if st.button('Predict'):
    if 'model' in locals():
        predict = prediction(air, process, rpm, torque, tool_wear, type)
        st.success(predict)
    else:
        st.error("Model is not loaded. Please check the error messages above.")

