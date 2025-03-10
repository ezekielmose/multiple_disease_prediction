# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 04:24:00 2025

@author: Alvine
"""

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import requests
import pickle
import joblib
import io

# Hearth Disease Model
hd_url = 'https://raw.githubusercontent.com/ezekielmose/multiple_disease_prediction/refs/heads/main/hd_model.pkl'


# Fetch the model file from GitHub
response = requests.get(hd_url)
response.raise_for_status()  # Ensure we notice bad responses (404, etc.)

# Load the model using pickle
hd_model = pickle.load(io.BytesIO(response.content))



# Stroke Model
stroke_url = 'https://raw.githubusercontent.com/ezekielmose/multiple_disease_prediction/refs/heads/main/strock_model_new2.pkl'


# Fetch the model file from GitHub
response = requests.get(stroke_url)
response.raise_for_status()  # Ensure we notice bad responses (404, etc.)

# Load the model using pickle
stroke_model = pickle.load(io.BytesIO(response.content))


# Side bar for navigation

with st.sidebar:
    selected = option_menu("Healthcare Models",
                           ["Heart Disease Prediction",
                            "Stroke Prediction"],default_index=0)

# Hearth Disease Prediction Page

if (selected == "Healthcare Models"):
    # set title
    st.title ("Hearth Disease Prediction model")
    
if (selected == "Stroke Prediction"):
    # set title
    st.title ("Strok Prediction model")
    
    
    # Apply custom CSS
    # Apply custom CSS to style the dropdown field and menu
    st.markdown("""
        <style>
         
            
            /* Style the dropdown menu */
            ul {
                background-color: smoky black !important;
            }
    
            /* Style dropdown options */
            li {
                color: white !important;
                font-weight: bold;
            }
    
            /* Change the hover effect on dropdown options */
            li:hover {
                background-color: grey !important;
                color: black !important;
            }
        </style>
    """, unsafe_allow_html=True)
     
    gender = int(st.selectbox(label = "What is the Gender (0 - Female and 1 - Male)", options = [1, 0]))

    age = st.number_input( "Select the Age using the + and - signs )", 
        min_value=1, 
        max_value=80, 
        step=1
                                       )

    #age = float(st.text_input("Enter the age", "0"))
    hypertension = int(st.text_input("Hypertension 0 for -ve and 1 for +ve", "0"))
    heart_disease = int(st.text_input("Heart disease 0 for has and 1 for not", "0"))
    ever_married = int(st.text_input("Ever married 0 for No and 1 for Yes", "0"))
    work_type = int(st.selectbox( label = "Work type 0-private. 1-self-employed, 2-children, 3-gov_job, and 4-never worked", options = [0, 1,2,3,4]))
    Residence_type = int(st.text_input("Residence type 0 for urban and 1 for rural", "0"))

    avg_glucose_level = st.number_input( "Select the avg_glucose_level using the + and - signs )", 
        min_value=100, 
        max_value=300, 
        step=1
                                       )
    
    #avg_glucose_level = float(st.text_input("Enter any value of avg_glucose_level as per the measurements", "0"))

    bmi = st.number_input( "Select the BMI value", 
        min_value=10, 
        max_value=97, 
        step=1
    )
    #bmi = float(st.text_input("Enter any value of BMI as per the measurements", "0"))

    smoking_status = st.number_input( "Smoking status (0: Never smoked, 1: Unknown, 2: Formerly smoked, 3: Smokes)", 
        min_value=0, 
        max_value=3, 
        step=1
    )
    
    #smoking_status = int(st.text_input("Smoking status 0 for never smoked, 1 for unknown, 2 for formerly smoked, 3 for smokes", "0"))
    
    if st.button('CLICK HERE TO PREDICT'):
        makeprediction = stroke_model.predict([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])
        output = round(makeprediction[0])  # Ensure it's an integer (0 or 1)
    
        if output == 0:
            st.success("The patient is **not at risk** of stroke.")
        else:
            st.warning("The patient **is at risk** of stroke. Please consult a doctor.")        

