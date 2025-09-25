import streamlit as st
import pandas as pd
import numpy as np
import time

import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
BASE_DIR = os.path.dirname(__file__)

#Initialize LableEncoder - Liver
with open(os.path.join(BASE_DIR,'labelencoder_liver.pkl'), 'rb') as file_en_liver:
    encoder_liver = pickle.load(file_en_liver)

#Initialize LableEncoder - Kidney
with open(os.path.join(BASE_DIR, 'labelencoder_kidney.pkl'), 'rb') as file_en_kidney:
    encoder_kidney = pickle.load(file_en_kidney)

#Initialize Scaler - Liver
with open(os.path.join(BASE_DIR,'scaler_liver.pkl'), 'rb') as scalerfile_liver:
    scaler_liver = pickle.load(scalerfile_liver)

#Initialize Scaler - Kidney
with open(os.path.join(BASE_DIR,'scaler_kidney.pkl'), 'rb') as scalerfile_kidney:
    scaler_kidney = pickle.load(scalerfile_kidney)

#Initialize Scaler - Parkinsons
with open(os.path.join(BASE_DIR,'scaler_parkins.pkl'), 'rb') as scalerfile_parkins:
    scaler_parkinsons = pickle.load(scalerfile_parkins)

# Loading the models - Liver
with open(os.path.join(BASE_DIR,'randomfor_liver.pkl'), 'rb') as file_liver:
    liver_model = pickle.load(file_liver)

# Loading the models - Kidney
with open(os.path.join(BASE_DIR,'randomfor_kidney.pkl'), 'rb') as file_kidney:
    kidney_model = pickle.load(file_kidney)

# Loading the models - Parkinsons
with open(os.path.join(BASE_DIR,'randomfor_parkins.pkl'), 'rb') as file_parkins:
    parkinsons_model = pickle.load(file_parkins)

# Streamlit UI name
st.set_page_config(page_title="Multiple Disease Prediction")

# Styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 34px;
        font-weight: bold;
        color: #4CAF50;
        text-shadow: 2px 2px 5px rgba(76, 175, 80, 0.4);
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        #color: #ddd;
        margin-bottom: 20px;
    }
    .stButton button {
        background: linear-gradient(to right, #4CAF50, #388E3C);
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton button:hover {
        background: linear-gradient(to right, #388E3C, #2E7D32);
    }
    .result-card {
        background: rgba(0, 150, 136, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0px 2px 8px rgba(0, 150, 136, 0.2);
    }
    .success-banner {
        background: linear-gradient(to right, #2E7D32, #1B5E20);
        color: white;
        padding: 15px;
        font-size: 18px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-top: 15px;
        box-shadow: 0px 2px 8px rgba(0, 150, 136, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title(" 📊 Disease Prediction Options")
disease = st.sidebar.radio("⚕️ Select Disease Prediction", ("🏠 Home","🫀 Liver Disease", "💧 Kidney Disease", "🧠 Parkinson’s Disease"))


if disease == '🏠 Home':
    st.markdown('<h1 class="main-title"> 🩺 Multiple Disease Prediction 🏥</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">⚕️A scalable AI-powered system for early disease detection 🤖, enabling faster ⚡, cost-effective 💰, and accurate ✅ healthcare decisions.</p>', unsafe_allow_html=True)



if disease == '🫀 Liver Disease':
    st.markdown(
    """
    <h3 style='text-align: center; color: #0B4242; text-shadow: 2px 2px 5px gray; word-spacing: 5px; border: 2px solid #333;
    border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); background-color: #F5F5F5;
    padding: 10px; margin: 15px;'>
        🫀 Liver Disease Prediction
    </h3>
    """,
    unsafe_allow_html=True)


    # Features
    liver_features = [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
        'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
        'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
        'Albumin_and_Globulin_Ratio'
    ]

    st.markdown(
    """
    <h3 style='text-align: center; color: #941F1F; font-style: italic; font-weight: normal; font-size: 24px;'>
        Enter Patient Details
    </h3>
    """,
    unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    user_data = {}

    default_values = {
        'Age': 40.0,
        'Total_Bilirubin': 1.0,
        'Direct_Bilirubin': 0.2,
        'Alkaline_Phosphotase': 100.0,
        'Alamine_Aminotransferase': 30.0,
        'Aspartate_Aminotransferase': 30.0,
        'Total_Protiens': 6.5,
        'Albumin': 3.5,
        'Albumin_and_Globulin_Ratio': 1.0
    }
    

    for i, feature in enumerate(liver_features):
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3

        if feature == "Gender":
            # Use same categories your encoder was trained on
            # Access the encoder for Gender
            gender_encoder = encoder_liver["Gender"]
            categories = list(gender_encoder.classes_)
            user_input = col.selectbox("Gender", categories)
            user_data[feature] = gender_encoder.transform([user_input])[0]
        else:
            user_data[feature] = col.number_input(
                f"{feature}", min_value=0.0,value=default_values.get(feature, 0.0), format="%.2f"

            )

    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    st.write("📊 Input Data for Prediction:")
    st.dataframe(input_df)

    left, mid, right = st.columns(3)

    if user_data and mid.button("🔍 Predict Disease"):

        numeric_values = [v for k, v in user_data.items() if k != "Gender"]
        if all(v == 0.0 for v in numeric_values):
            st.info("⚠️ Cannot predict. Please enter realistic patient data.")
        else:
            input_data = input_df[liver_features].values 

            input_scaled = scaler_liver.transform(input_data)

            prediction = liver_model.predict(input_scaled)

            if prediction[0]==0:
                st.balloons()
                st.success("🎉 You are doing well! No disease detected at this time.")
            else:
                st.warning("🩺 Attention: Signs of disease risk detected. Follow up with your doctor for detailed diagnosis.")



if disease == '💧 Kidney Disease':
    st.markdown(
    """
    <h3 style='text-align: center; color: #0A4461; text-shadow: 2px 2px 5px gray; word-spacing: 5px; border: 2px solid #333;
    border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); background-color: #F5F5F5;
    padding: 10px; margin: 15px;'>
        💧 Kidney Disease
    </h3>
    """,
    unsafe_allow_html=True)


    # Features
    kidney_features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
       'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
       'appet', 'pe', 'ane']

    st.markdown(
    """
    <h3 style='text-align: center; color: #941F1F; font-style: italic; font-weight: normal; font-size: 24px;'>
        Enter Patient Details
    </h3>
    """,
    unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    user_data = {}

    for i, feature in enumerate(kidney_features):
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3

        if feature in encoder_kidney:
            feature_encoder = encoder_kidney[feature]
            categories = list(feature_encoder.classes_)
            user_input = col.selectbox(feature, categories)
            user_data[feature] = feature_encoder.transform([user_input])[0]
        else:
            user_data[feature] = col.number_input(
                f"{feature}", min_value=0.0, format="%.2f"
            )

    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    st.write("📊 Input Data for Prediction:")
    st.dataframe(input_df)

    left, mid, right = st.columns(3)

    if user_data and mid.button("🔍 Predict Disease"):

        input_data = input_df[kidney_features].values 

        input_scaled = scaler_kidney.transform(input_data)


        prediction = kidney_model.predict(input_scaled)

        if prediction[0]==0:
            st.balloons()
            st.success("✅ Your health looks good! Keep maintaining a healthy lifestyle.")
        else:
            st.warning("🩺 Attention: Signs of disease risk detected. Follow up with your doctor for detailed diagnosis.")


if disease == "🧠 Parkinson’s Disease":
    st.markdown(
    """
    <h3 style='text-align: center; color: #400733; text-shadow: 2px 2px 5px gray; word-spacing: 5px; border: 2px solid #333;
    border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); background-color: #F5F5F5;
    padding: 10px; margin: 15px;'>
        🧠 Parkinson’s Disease
    </h3>
    """,
    unsafe_allow_html=True)


    # Features
    parkins_features = ['MDVP_Fo_Hz', 'MDVP_Fhi_Hz', 'MDVP_Flo_Hz', 'MDVP_Jitter_%',
       'MDVP_Jitter_Abs', 'MDVP_RAP', 'MDVP_PPQ', 'Jitter_DDP', 'MDVP_Shimmer',
       'MDVP_Shimmer_dB', 'Shimmer_APQ3', 'Shimmer_APQ5', 'NHR', 'HNR', 'PPE']

    st.markdown(
    """
    <h3 style='text-align: center; color: #941F1F; font-style: italic; font-weight: normal; font-size: 24px;'>
        Enter Patient Details
    </h3>
    """,
    unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    user_data = {}

    for i, feature in enumerate(parkins_features):
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3

        user_data[feature] = col.number_input(
            f"{feature}", min_value=0.0, format="%.2f")

    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    st.write("📊 Input Data for Prediction:")
    st.dataframe(input_df)

    left, mid, right = st.columns(3)

    if user_data and mid.button("🔍 Predict Disease"):

        input_data = input_df[parkins_features].values 

        input_scaled = scaler_parkinsons.transform(input_data)


        prediction = parkinsons_model.predict(input_scaled)

        if prediction[0]==0:
            st.balloons()
            st.success("💼 Great news! You are healthy and no signs of disease were detected.")
        else:
            st.warning("🩺 Attention: Signs of disease risk detected. Follow up with your doctor for detailed diagnosis.")

