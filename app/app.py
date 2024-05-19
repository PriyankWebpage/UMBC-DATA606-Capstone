import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image




with open('app/finalized_model.pkl', 'rb') as file:
    model = pickle.load(file)


def predict_note_authentication(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, Type):

   
    prediction = model.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, Type]])
    print(prediction)
    return prediction



def main():
    
    st.title("Wine Quality Classifier Web App")
    css_code_image = f"""
    <style>
    .stApp {{
        background-image: url("https://wallpapercave.com/wp/wp3982250.jpg");
        background-size: cover;
        background-position: right;
        background-repeat: no-repeat;
    }}
    </style>
    """
    

    fixed_acidity, volatile_acidity = st.columns(2)

    with fixed_acidity:
        fixed_acidity = st.text_input("Fixed Acidity",align_text='left')

    with volatile_acidity:
      volatile_acidity = st.text_input("Volatile Acidity")

    
    citric_acid, residual_sugar = st.columns(2) 
    
    with citric_acid:
        citric_acid = st.text_input("Citric Acid")

    with residual_sugar:
     residual_sugar = st.text_input("Residual Sugar")

    
    chlorides, free_sulfur_dioxide = st.columns(2) 
    
    with chlorides:
        chlorides = st.text_input("Chlorides")

    with free_sulfur_dioxide:
     free_sulfur_dioxide = st.text_input("Free Sulfur Dioxide")
    
    total_sulfur_dioxide, density = st.columns(2) 
    
    with total_sulfur_dioxide:
        total_sulfur_dioxide = st.text_input("Total Sulfur Dioxide") 

    with density:
     density = st.text_input("Density")

    pH, sulphates = st.columns(2) 
    
    with pH:
        pH = st.text_input("pH")

    with sulphates:
        sulphates = st.text_input("Sulphates")
    
    
    alcohol, Type = st.columns(2) 
    
    with alcohol:
        alcohol = st.text_input("Alcohol Concentration")

    with Type:
        Type  = st.selectbox('Choose wine type', ('Red', "White",))
        Type = 1 if Type == 'Red' else 0

    result=""
    if st.button("Predict"):
        result=predict_note_authentication(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, Type)
    
    if result == 1:
        result = 'Good' 
    if result == 0:
       result = 'Poor'
    st.success('The quality of wine is {}'.format(result))

if __name__=='__main__':
    main()
