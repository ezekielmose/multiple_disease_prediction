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
hd_url = 'https://raw.githubusercontent.com/ezekielmose/Machine-Learning/refs/heads/main/trained_model.sav'


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
                           ["Heart Disease Prediction", "Stroke Prediction"] , default_index=0)

# Hearth Disease Prediction Page

if (selected == "Heart Disease Prediction"):
    # set title
    # st.title ("Hearth Disease Prediction model")

    def hearth_disease_prediction(input_data):

        input_data_as_numpy_array= np.array(input_data)
    # reshaping the array for predicting 
    
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # instead of 'model' we use loaded_model
        prediction = hd_model.predict(input_data_reshaped)


        if prediction [0] == 0:
            return "The Person Does not have a Heart Disease"
        else:
            return "The Person has Heart Disease" # insted of print change to return  
    
    # Streamlit library to craete a user interface
    def main():
        
        # Interface title
        st.title("Heart Disease Prediction Machine Learning Model")
        
        #getting the input data from the user  
        age = st.text_input("Enter the Patient's Age 15 - 80")
        sex = st.text_input("Enter the Patient's Gender (0 [F] or 1[M])")
        Chest_Pain = st.text_input("Chest Pain level (0,1,2 or 3)")
        Blood_Pressure= st.text_input("The Blood Pressure(mm Hg)level (94-200) ")
        cholestoral = st.text_input("Cholestoral Level (mg/dl) (131 -290)")
        Fasting_Blood_Sugar = st.text_input("Patient's Fasting Blood Sugar (0,1)")
        resting_electrocardiographic = st.text_input("Electrocardiographic level (0, 1 or 2)")
        Maximum_Heart_Rate= st.text_input("Maximum Heart Rate (99 - 162)")
        Excersize_Includes = st.text_input("Enter the Patient's Excersize_Includes")
        ST_Depression = st.text_input("Patient's ST Depression [ECG or EKG] (0.0 - 4.4)")
        Slope_of_Excersize	 = st.text_input("Patient's Slope of Excersize (0,1 or 2)")
        Number_of_vessels = st.text_input("Number of vessels (0, 1,2,3 or 4)")
        Thalassemia = st.text_input("Thalassemia (1,2,3 or 4)")
    
    
        ## Numeric conversion
        # Convert inputs to numeric using pd.to_numeric or float conversion
        age = pd.to_numeric(age, errors='coerce') # errors ='coerce' - tells pandas to force any non-convertible values like text or invalid numbers to NAN
        sex = pd.to_numeric(sex, errors='coerce')
        Chest_Pain = pd.to_numeric(Chest_Pain, errors='coerce')
        Blood_Pressure = pd.to_numeric(Blood_Pressure, errors='coerce')
        cholestoral = pd.to_numeric(cholestoral, errors='coerce')    
        Fasting_Blood_Sugar = pd.to_numeric(Fasting_Blood_Sugar, errors='coerce')
        resting_electrocardiographic = pd.to_numeric(resting_electrocardiographic, errors='coerce')
        Maximum_Heart_Rate = pd.to_numeric(Maximum_Heart_Rate, errors='coerce')
        Excersize_Includes = pd.to_numeric(Excersize_Includes, errors='coerce')
        ST_Depression = pd.to_numeric(ST_Depression, errors='coerce')
        Slope_of_Excersize = pd.to_numeric(Slope_of_Excersize, errors='coerce')
        Number_of_vessels = pd.to_numeric(Number_of_vessels, errors='coerce')
        Thalassemia = pd.to_numeric(Thalassemia, errors='coerce')
    
        # code for prediction ### refer to prediction function
        diagnosis = '' # string tha contains null values whose values are stored in the prediction
        
        # creating  a prediction button
        if st.button("PREDICT"):
            diagnosis = hearth_disease_prediction([age,sex,Chest_Pain,Blood_Pressure,cholestoral])
        st.success(diagnosis)
        
     
    # this is to allow our web app to run from anaconda command prompt where the cmd takes the main() only and runs the code
    if __name__ == '__main__':
        main()

    
if (selected == "Stroke Prediction"):
    # set title
    st.title ("Stroke Prediction model")
    
    
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

