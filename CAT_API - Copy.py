import numpy as np
import numpy as np
import pandas as pd
#import lux
#lux.logger = True
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import scipy
from math import sqrt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import time
import random
%matplotlib inline
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image


df = pd.read_excel('C:/Users/eng_a/Desktop/Masters Kuniv/Third semester/Thesis_app/Thesis_progress/comparing_handling_missing_values/Interpolation_Average_2013_2015.xlsx')
df = df.drop(['Unnamed: 0'], axis =1)
df = df.set_index('MeasurementDateTime')
from sklearn import preprocessing
x_1= df[['WS-Hour','Temp-Hour', 'SR-Hour' ,'RH-Hour', 'NO2','SO2','WD-Hour']]
y_2=df[['O3']]
#y_2 = np.ravel(df[['O3']])
scaled_inputs = preprocessing.scale(x_1)
x_train, x_val, y_train, y_val = train_test_split(scaled_inputs,y_2,test_size=0.2,shuffle=True,random_state = 42)
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs,y_2,test_size=0.1,shuffle=True,random_state = 42)
from catboost import CatBoostRegressor
model=CatBoostRegressor(learning_rate= 0.05, n_estimators=600, depth=16, l2_leaf_reg = 0.5, loss_function='RMSE')
model.fit(x_train, y_train)
import pickle
pickle_out = open("model_cat.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("model_cat.pkl","rb")
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
   
    prediction=model_cat.predict([[WD_Hour,WS_Hour,Temp_Hour,SR_Hour,RH_Hour,SO2,NO2]])
    print(prediction)
    return prediction



def main():
    st.title("Ozone Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Ozone Predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    WS_Hour = st.text_input("WS-Hour","Type Here")
    Temp_Hour = st.text_input("Temp-Hour","Type Here")
    SR_Hour = st.text_input("SR-Hour","Type Here")
    RH_Hour = st.text_input("RH-Hour","Type Here")
    NO2 = st.text_input("NO2","Type Here")
    SO2 = st.text_input("SO2","Type Here")
    WD_Hour = st.text_input("WD-Hour","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_ozone(WS_Hour,Temp_Hour,SR_Hour,RH_Hour,NO2,SO2,WD_Hour)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()