import os
import shutil
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
# import altair as alt
#import mpld3
import streamlit.components.v1 as components
#from flasgger import Swagger
import streamlit as st 
from PIL import Image
import base64
from pandas_profiling import ProfileReport
import warnings
warnings.filterwarnings("ignore")
# from streamlit_pandas_profiling import st_profile_report
import json
# button style
m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: green;
        }
        </style>""", unsafe_allow_html=True)

#b = st.button("test")

#background theme
#def add_bg_from_local(image_file):
   # with open(image_file, "rb") as image_file:
  #      encoded_string = base64.b64encode(image_file.read())
 #   st.markdown(
   # f"""
   # <style>
  #  .stApp {{
 #       background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#        background-size: cover
#   }}
#    </style>
#   """,
#    unsafe_allow_html=True
#   )
#add_bg_from_local('AI.png')  
    

# Web App Title
html_temp = """
    <div style="background-color:orange;padding:10px">
    <h3 style="color:black;text-align:center;font-size:40px"><b>Aritar ML App</b></h3>
    </div>
    
    """
st.markdown(html_temp,unsafe_allow_html=True)
st.markdown('''
## **PCS 180 Max Coil DP Predictive Analysis**
**Credit:** App built in `Python` + `Streamlit`
''')

# load the stored ml models
file_name = "rf_model_Max_Coil_DP"
folder_name = "rf_model_Max_Coil_DP"
parent_dir_path = "D:/PCS 180/ml_models/PCS_180" 
model_path = os.path.join(parent_dir_path, folder_name)

with open(model_path + '/' + file_name, 'rb') as f: 
    model = pickle.load(f)    

# Upload raw data
html_temp_side_bar = """
    <div style="background-color:green;padding:10px">
    <h5 style="color:white;text-align:center;font-size:15px"><b>Upload the raw data to obtain basic information 
    of the dataset</b></h5>
    </div>
    
    """
with st.sidebar.markdown(html_temp_side_bar, unsafe_allow_html=True):
    uploaded_raw_file = st.sidebar.file_uploader("Input Raw", type=["xlsx"])

if uploaded_raw_file is not None:
    def load_raw_file():
        file = pd.read_excel(uploaded_raw_file)
        return file
    df = load_raw_file()
    pr = ProfileReport(df , explorative = True)
    st.header('**Pandas profile report**')
    # st_profile_report(pr)
    

    
# Upload Excel data
html_temp_side_bar = """
    <div style="background-color:green;padding:10px">
    <h5 style="color:white;text-align:center;font-size:15px"><b>Upload Excel files to predict Max_Coil_DP</b></h5>
    </div>
    
    """
with st.sidebar.markdown(html_temp_side_bar, unsafe_allow_html=True):
    uploaded_file = st.sidebar.file_uploader("Input Excel", type=["xlsx"])

def welcome():
    return "Welcome All"
        


def predict_Max_Coil_DP(TOTAL_FEED,
                    TOTAL_DS,
                    DS_FEED_RATIO,
                    Avg_COT,
                    Draft_Pressure,
                    Excess_Oxygen,
                    Average_Cracked_Gas_Temp,
                    Average_USX_Outlet):
    
    
    """Let's Prdict the Max_Coil_DP produce in Furnace 180
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: TOTAL_FEED
    in: query
    type: number
    required: true
      - name: Total_DS
    in: query
    type: number
    required: true
      - name: DS_FEED_RATIO
    in: query
    type: number
    required: true
      - name: Avg_COT
    in: query
    type: number
    required: true
      - name: Draft_Pressure
    in: query
    type: number
    required: true
      - name: Excess_Oxygen
    in: query
    type: number
    required: true
      - name: Average_Cracked_Gas_Temp
    in: query
    type: number
    required: true
      - name: Average_USX_Outlet
    in: query
    type: number
    required: true
      responses:
    200:
        description: The output values
    
    """
   
    prediction=model.predict([[TOTAL_FEED,
                        TOTAL_DS,
                        DS_FEED_RATIO,
                        Avg_COT,
                        Draft_Pressure,
                        Excess_Oxygen,
                        Average_Cracked_Gas_Temp,
                        Average_USX_Outlet]])
    print(prediction)
    print(prediction.dtype)
    return prediction
predict_Max_Coil_DP(24.99,10.74,0.42,806.32,-5.53,1.17,599.73,598.27)
# method to load excel files
df = pd.read_excel("D:/PCS 180/val data/actual.xlsx")
prediction = model.predict(df)
pred_df = pd.DataFrame({'Predicted Max_Coil_DP' : prediction})
print(pred_df)
print(pred_df.dtypes)
def load_excel(file):
    excel = pd.read_excel(file)
    return excel

def main():
    
    if uploaded_file is not None:
        df = load_excel(uploaded_file)
        if st.button('Press to predict Max_Coil_DP using Input Data'):
            st.title('**Input DataFrame**')
            st.write(df)
            st.title('**Predicted Max_Coil_DP**')
            prediction = model.predict(df)
            pred_df = pd.DataFrame({'Predicted Max_Coil_DP' : prediction})
            print(pred_df)
            st.write(pred_df)
            file_path = "D:/PCS 180"
            folder_name = "val data"
            path = os.path.join(file_path, folder_name)
            df1 = pd.read_excel(path + "/Actual_max_coil_dp.xlsx")
            st.title('**Actual vs Predicted**') 
            chart = pd.concat([df1 , pred_df],axis = 1)
            line = st.line_chart(chart)
    html_temp_pred_single_instances = """
    <div style="background-color:green;padding:10px">
    <h5 style="color:white;text-align:center;font-size:15px"><b>Below we can also predict Max_Coil_DP for a single instance </b></h5>
    </div>
    
    """
    # st.info(html_temp_pred_single_instances, unsafe_allow_html=True)
    st.info("Below we can also predict Max_Coil_DP for a single instance...")
    html_temp1 = """
        <div style="background-color:orange;padding:10px">
        <h3 style="color:black;text-align:left;font-size:20px"><b>Select the parameter details below</b></h3>
        </div>
        
        """

    st.markdown(html_temp1,unsafe_allow_html=True)
    #st.subheader("Enter parameters details below")  
    # creating a form fields
    with st.form(key="form1"):
        TOTAL_FEED = st.slider("Enter TOTAL_FEED", 
                                            min_value=0.00,
                                            max_value=100.0, step=0.001)
        #st.write(Naphtha_Feed_rate_Total)
        
        TOTAL_DS = st.slider("Enter Total_DS", 
                                            min_value=0.00,
                                            max_value=100.0, step=0.001)
        #st.write(Total_DS_rate_to_individual_COil)
        
        DS_FEED_RATIO = st.slider("Enter DS_FEED_RATIO", 
                                            min_value=0.00,
                                            max_value=10.0, step=0.001)
        #st.write(Total_DS_rate_to_individual_COil)
        
        Avg_COT = st.slider("Enter Avg_COT", 
                                            min_value=500.00,
                                            max_value=1500.0, step=0.001)
        #st.write(COT_Avg)
        Draft_Pressure = st.slider("Enter Draft Pressure", 
                                            min_value=-6.00,
                                            max_value=5.00, step=0.001)
        #st.write(Q_Wall)
        
        Excess_Oxygen = st.slider("Enter Excess Oxygen", 
                                            min_value=0.00,
                                            max_value=10.00, step=0.001)
        #st.write(Q_Ground)
        
        Average_Cracked_Gas_Temp = st.slider("Enter Average_Cracked_Gas_Temp", 
                                            min_value=500.00,
                                            max_value=1000.00, step=0.001)
        #st.write(O2)
        
        Average_USX_Outlet = st.slider("Enter Average_USX_Outlet", 
                                            min_value=500.00,
                                            max_value=1000.00, step=0.001)
        #st.write(Draft_Pressure)  
        
        
        results = st.form_submit_button(label="Predict")
    
    
    with st.expander("Results"):
        if results:
            arr_ls = [[TOTAL_FEED,
                        TOTAL_DS,
                        DS_FEED_RATIO,
                        Avg_COT,
                        Draft_Pressure,
                        Excess_Oxygen,
                        Average_Cracked_Gas_Temp,
                        Average_USX_Outlet]]
            Max_Coil_DP = model.predict(arr_ls)
                
            
            st.dataframe({'Predicted MAX CIP': Max_Coil_DP})
            #st.success('The predictive Max CIP : {}'.format(result))

if __name__=='__main__':
    main()