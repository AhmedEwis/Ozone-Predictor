import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("model_cat_new.pkl","rb")
model_cat=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_ozone(WS_Hour,Temp_Hour,SR_Hour,RH_Hour,NO2,SO2,WD_Hour):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=model_cat.predict([[WD_Hour,WS_Hour,Temp_Hour,SR_Hour,RH_Hour,NO2]])
    print(prediction)
    return prediction



def main():
    st.title("Welcome All to the Ozone Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Ozone Predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    WD_Hour = st.text_input("WD-Hour","Type Here")
    WS_Hour = st.text_input("WS-Hour","Type Here")
    Temp_Hour = st.text_input("Temp-Hour","Type Here")
    SR_Hour = st.text_input("SR-Hour","Type Here")
    RH_Hour = st.text_input("RH-Hour","Type Here")
    NO2 = st.text_input("NO2","Type Here")
    #SO2 = st.text_input("SO2","Type Here")
    
    result=""
    if st.button("Predict"):
        result=predict_ozone(WS_Hour,Temp_Hour,SR_Hour,RH_Hour,NO2,WD_Hour)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Author: Ahmed Ewis")
        st.text("Supervised By: Dr. Fahad Al Fadli")

if __name__=='__main__':
    main()