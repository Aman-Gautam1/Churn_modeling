import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder
import pandas as pd
import pickle

# load the trained model 

model = tf.keras.model.load_model('model.h5')

#load the encoders,scalers

with open('onehot_geo.pkl','rb') as file:
    onehot_geo =pickle.load(file)

with open("label_gender.pkl",'rb') as file:
    label_gender = pickle.load(file)

with open("scaler.pkl",'rb') as file:
    scaler = pickle.load(file)


#streamlit app
st.title("Customer Churn Prediction")

# user inputs

geography = st.selectbox("Geography",onehot_geo.categories_[0])
gender = st.selectbox('Gender',label_gender.classes_)
age = st.slider('Age',18,90)
balance = st.number_input('Balance')
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Numeber of Products',1,4)
has_cr_card = st.selectbox("Has credit card",[0,1])
is_active_member = st.selectbox("Is active member ",[0,1])


input_data = {
    'CreditScore':[credit_score],
    'Gender':[label_gender.transform([gender])[0]],
    
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]

}

geo_encoded = onehot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded , columns = onehot_geo.get_feature_names_out(["Geography"]))


input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis = True)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    print("The customer is likely to churn")
else:
    print("The customer is not likely to churn")