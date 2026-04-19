import streamlit as st
import pandas as pd
import numpy as nnp
import pickle
import base64
import os
import xgboost as xgb
def get_binary_file_downloader_html(df):
    csv=df.to_csv(index=False)
    b64=base64.b64encode(csv.encode()).decode()
    href=f'<a href="data:file/csv;base,{b64}" download="predictions.csv">Download predictions CSV</a>'
    return href
st.title("Heart disease predictor")
tab1,tab2=st.tabs(['Predict','Bulk predict'])
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
    Sex=0 if Sex=="Male" else 1
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
    algonames=['Decision trees','Logistic regression','Random forest','Support vector machine','XGBoost']
    modelnames=['decisiontree.pkl','logisticR.pkl','randomforest.pkl','svm.pkl','xgboost.pkl']
    predictions=[]
    def predict_heart_disease(data):
        for modelname in modelnames:
            model=pickle.load(open(modelname,'rb'))
            prediction=model.predict(data)
            predictions.append(prediction)
        return predictions
    if st.button("Submit"):
        st.subheader("Results")
        result=predict_heart_disease(input_data)
        for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0]==0:
                st.write("No heart disease detected")
            else:
                st.write("Heart disease detected")
 
 
with tab2:
    st.title("Upload csv file")
    try:
        model = pickle.load(open("xgb_model.pkl", "rb"))
        columns = pickle.load(open("columns.pkl", "rb"))
    except:
        st.error("Model files not found. Run train_model.py first.")
        st.stop()
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)
        except:
            st.error("Invalid CSV file.")
            st.stop()
        expected_columns = [
            'Age','Sex','ChestPainType','RestingBP','Cholesterol',
            'FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope'
        ]
        if not set(expected_columns).issubset(input_data.columns):
            st.error("CSV does not have required columns.")
            st.stop()
        if input_data.isnull().values.any():
            st.error("CSV contains missing values.")
            st.stop()
        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)
        predictions = model.predict(input_encoded)
        input_data["Prediction"] = predictions
        input_data["Result"] = input_data["Prediction"].map({
            1:"Heart Disease",
            0:"No Heart Disease"
        })
        st.subheader("Predictions")
        st.dataframe(input_data)
        st.download_button(
            label="Download Predictions",
            data=input_data.to_csv(index=False),
            file_name="Predicted_Heart.csv",
            mime="text/csv"
        )
    else:
        st.info("Please upload a CSV file to get predictions.")

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
