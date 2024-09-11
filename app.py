# Machine-Learning-Project-3-Navtcc
import streamlit as st
import numpy as np
import tensorflow as tf

# Load the Keras model
model_path = 'path_to_your_model.h5'
model = tf.keras.models.load_model(model_path)

# Get the input shape from the model
input_shape = model.input.shape[1:]
num_features = np.prod(input_shape)

# Function to process user inputs and make predictions
def predict(inputs):
    inputs = np.array(inputs).reshape((1, *input_shape))
    prediction = model.predict(inputs)
    return prediction

# Streamlit app
st.title('RandomForestRegressor Model Prediction App')

# Collect user inputs
st.write(f'Please enter values for {num_features} inputs:')
inputs = []
for i in range(num_features):
    value = st.number_input(f'Input {i+1}', value=0.0)
    inputs.append(value)

if st.button('Predict'):
    # Predict and display results
    prediction = predict(inputs)
    st.write('Prediction:')
    st.write(prediction)