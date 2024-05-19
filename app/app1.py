import streamlit as st
import numpy as np
import pickle

# Load the model
with open('app/finalized_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Prediction function
def predict_note_authentication(inputs):
    # Assuming model expects numpy array for prediction
    prediction = model.predict([inputs])
    return prediction[0]

def main():
    st.title("Wine Quality Classifier Web App")

    # CSS to align everything to the left
    # css_code = """
    # <style>
    # html, body, [class*="View"] {
    #     margin: 0px !important;
    #     padding: 0px !important;
    # }
    # .stApp {
    #     background-image: url("https://wallpapercave.com/wp/wp3982250.jpg");
    #     background-size: cover;
    #     background-position: right;
    #     background-repeat: no-repeat;
    # }
    # .css-1d391kg, .css-18e3th9 {
    #     padding-left: 100;
    #     padding-right: -100;
    #     margin-left: 0!important;
    #     width: 100%!important;
    #     max-width: none!important;
    # }
    # </style>
    # """
    # st.markdown(css_code, unsafe_allow_html=True)

    # Input fields
    cols = st.columns(2)
    inputs = []

    # Labels for input fields
    labels = ["Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar",
              "Chlorides", "Free Sulfur Dioxide", "Total Sulfur Dioxide", "Density",
              "pH", "Sulphates", "Alcohol Concentration", "Choose wine type (Red or White)"]
    
    # Generate text input fields
    for i, label in enumerate(labels):
        with cols[i % 2]:
            if label.endswith("type (Red or White)"):
                # Select box for wine type
                wine_type = st.selectbox(label, ('Red', 'White'))
                wine_type = 1 if wine_type == 'Red' else 0
                inputs.append(wine_type)
            else:
                # Regular input
                value = st.text_input(label)
                inputs.append(value)

    if st.button("Predict"):
        # Convert text inputs to float where necessary except for the last item (wine type)
        float_inputs = [float(x) if i != len(inputs) - 1 else x for i, x in enumerate(inputs[:-1])]
        float_inputs.append(inputs[-1])  # Append wine type which is already in binary form
        result = predict_note_authentication(float_inputs)
        quality = 'Good' if result == 1 else 'Poor'
        st.success(f'The quality of wine is {quality}')

if __name__ == '__main__':
    main()
