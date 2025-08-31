import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# Load the model and preprocessing objects
model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

# App title
st.title("Customer Churn Prediction")
st.write("Predict whether a customer will churn based on their features.")

# Input form
st.subheader("Customer Details")
geography = st.selectbox("Geographic Location", onehot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder.classes_)
age = st.slider("Age", 18, 100, 30)
balance = st.number_input("Account Balance", value=0.0)
credit_score = st.slider("Credit Score", 300, 850, 650)
estimated_salary = st.number_input("Estimated Salary", value=50000.0)
tenure = st.slider("Tenure (years)", 0, 10, 2)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.radio("Has Credit Card?", ["No", "Yes"])
is_active_member = st.radio("Is Active Member?", ["No", "Yes"])

# Convert Yes/No to 1/0
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

# Predict button
if st.button("Predict Churn"):
    # Prepare input data
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # One-hot encode geography
    geo_encoded = onehot_encoder.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale input data and make prediction
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Display results
    st.subheader("Prediction Results")
    st.write(f"Churn Probability: {prediction_proba:.2%}")
    
    if prediction_proba > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")