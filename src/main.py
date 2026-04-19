import streamlit as st
import pandas as pd
import numpy as nnp
import pickle
import base64
import os
st.title("Heart disease predictor")
tab1,tab2,tab3=st.tabs(['Predict','Bulk predict','Model information'])
with tab1:
    Age=st.number_input("Age",min_value=0,max_value=150)
    Sex=st.selectbox("Sex",["Male","Female"])
    ChestPainType=st.selectbox("Chest pain type",["Typical angina","Atypical angina","Non anginal pain","Asymmptomatic"])
    RestingBP=st.number_input("Resting blood pressure",min_value=0,max_value=300)
    Cholesterol=st.number_input("Serum cholesterol",min_value=0)
    FastingBS=st.selectbox("Fasting blood sugar",["<=120mg/dl",">120mg/dl"])
    RestingECG=st.selectbox("Resting ecg results:",["Normal",'ST-T wave abnormality',"Left ventricular hypertrophy"])
    MaxHR=st.number_input("Maximum heart rate achieved",min_value=0.0,max_value=10.0)
    ExerciseAngina=st.selectbox("Exercise induced angina",["Yes","No"])
    Oldpeak=st.number_input("Oldpeak(ST Depression)",min_value=0.0,max_value=10.0)
    ST_Slope=st.selectbox("Slope of peak exercise st segment",["Upsloping","Flat","Downsloping"])
    Sex=0 if Sex==Male else 1
    ChestPainType=["Atypical angina","Non anginal pain","Asymptomatic","Typical angina"].index(ChestPainType)
    FastingBS=1 if FastingBS==">120mg/dl" else 0
    RestingECG=["Normal","ST-T wave abnormality","Left ventricular hypertrophy"].index(RestingECG)
    ExerciseAngina=1 if ExerciseAngina=="Yes" else 0
    ST_Slope=["Upsloping","Flat","Downsloping"].index(ST_Slope)
    input_data=pd.DataFrame({
        'Age':[Age],
        'Sex':[Sex],
        'ChestPainType':[ChestPainType],
        'RestingBP':[RestingBP],
        'Cholesterol':[Cholesterol],
        'FastingBS':[FastingBS],
        'RestingECG':[RestingECG],
        'MaxHR':[MaxHR],
        'ExerciseAngina':[ExerciseAngina],
        'Oldpeak':[Oldpeak],
        'ST_Slope':[ST_Slope]
    })