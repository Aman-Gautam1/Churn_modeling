import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

# Load the trained model 
model = tf.keras.models.load_model('model.h5')

# Load the encoders, scalers
with open('onehot_geo.pkl', 'rb') as file:
    onehot_geo = pickle.load(file)

with open("label_gender.pkl", 'rb') as file:
    label_gender = pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# User inputs
geography = st.selectbox("Geography", onehot_geo.categories_[0])
gender = st.selectbox('Gender', label_gender.classes_)
age = st.slider('Age', 18, 90)
balance = st.number_input('Balance')
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox("Has credit card", [0, 1])
is_active_member = st.selectbox("Is active member", [0, 1])

# Prepare input data
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

# Encode Geography and merge with input data
geo_encoded = onehot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geo.get_feature_names_out(["Geography"]))

input_data = pd.DataFrame(input_data)
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display the result
if prediction_proba > 0.5:
    st.write("The customer is **likely to churn**.")
else:
    st.write("The customer is **not likely to churn**.")
