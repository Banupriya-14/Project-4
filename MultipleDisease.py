import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

#Initialize LableEncoder - Liver
with open(r'E:\MDTM40\Project4_MultipleDisease\labelencoder_liver.pkl', 'rb') as file_en_liver:
    encoder_liver = pickle.load(file_en_liver)

#Initialize LableEncoder - Kidney
with open(r'E:\MDTM40\Project4_MultipleDisease\labelencoder_kidney.pkl', 'rb') as file_en_kidney:
    encoder_kidney = pickle.load(file_en_kidney)

#Initialize Scaler - Liver
with open('E:\MDTM40\Project4_MultipleDisease\scaler_liver.pkl', 'rb') as scalerfile_liver:
    scaler_liver = pickle.load(scalerfile_liver)

#Initialize Scaler - Kidney
with open('E:\MDTM40\Project4_MultipleDisease\scaler_kidney.pkl', 'rb') as scalerfile_kidney:
    scaler_kidney = pickle.load(scalerfile_kidney)

#Initialize Scaler - Parkinsons
with open('E:\MDTM40\Project4_MultipleDisease\scaler_parkins.pkl', 'rb') as scalerfile_parkins:
    scaler_parkinsons = pickle.load(scalerfile_parkins)

# Loading the models - Liver
with open(r'E:\MDTM40\Project4_MultipleDisease\randomfor_liver.pkl', 'rb') as file_liver:
    liver_model = pickle.load(file_liver)

# Loading the models - Kidney
with open(r'E:\MDTM40\Project4_MultipleDisease\randomfor_kidney.pkl', 'rb') as file_kidney:
    kidney_model = pickle.load(file_kidney)

# Loading the models - Parkinsons
with open(r'E:\MDTM40\Project4_MultipleDisease\randomfor_parkins.pkl', 'rb') as file_parkins:
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

st.markdown('<h1 class="main-title"> 🩺 Multiple Disease Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">⚕️A scalable AI-powered system for early disease detection 🤖, enabling faster ⚡, cost-effective 💰, and accurate ✅ healthcare decisions.</p>', unsafe_allow_html=True)

st.sidebar.title(" 📊 Disease Prediction Options")
disease = st.sidebar.radio("⚕️ Select Disease Prediction", ("Liver Disease", "Kidney Disease", "Parkinson's Disease"))
