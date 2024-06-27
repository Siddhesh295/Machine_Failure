import os
import joblib
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
model_path = 'https://github.com/Siddhesh295/Machine_Failure/blob/7296e1d835ae9dc5e21d4fb60c1529e6f1fcd533/Model/model.pkl'
if not os.path.exists(model_path):
    st.error(f"Model file not found at path: {model_path}")
    st.stop()

# Load the model
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

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
        'Air temperature [K]': [air],
        'Process temperature [k]': [process],
        'Rotational speed [rpm]': [rpm],
        'Torque [nm]': [torque],
        'Tool wear [min]': [tool_wear],
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
