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
    st.title ("Stroke Prediction model")
