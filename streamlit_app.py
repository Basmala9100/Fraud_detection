import streamlit as st
import pickle
import xgboost
import pandas as pd
from xgboost import XGBClassifier

# Load the pickled fraud detection model
def load_model(model_path):
    model = XGBClassifier()
    model.load_model(model_path)
    return model 

stander = pickle.load(open("stander.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

categorical_features = ['Merchant City','Merchant State','Zip', 'MCC','Errors?','Use Chip']
# Function to preprocess data (replace this with your actual data preprocessing function)
def preprocess_data(data):
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    data_copy = data.copy()

    for feature in categorical_features:
        if feature in data_copy.columns:
            data_copy[feature] = encoder.transform(data_copy[feature])
    
    data_copy = stander.transform(data_copy)  # Corrected 'transform' method call

    return data_copy

def on_button_click():
    st.write("Button clicked!")

st.title('Fraud Detection in Financial Transactions')
st.info('For Predect if Fraud or not')
# Dropdown menu for users to select different transaction types
transaction_type = st.selectbox('Select transaction type:', ['Predict with Email', 'Predict with Transaction credit card'])


if transaction_type == 'Predict With Email':
    st.write('Fara8')
elif transaction_type == 'Predict with Transaction credit card' :
    Card = st.selectbox('Card', [i for i in range(9)])
    Year = st.selectbox('Year', [i for i in range(1992, 2020)])
    Month = st.selectbox('Month', [i for i in range(1, 13)])
    Day = st.selectbox('Day', [i for i in range(1, 32)])
    Amount = st.number_input('Amount')
    Use_Chip = st.selectbox('Use Chip', ['Swipe Transaction', 'Chip Transaction'])
    Merchant_City = st.text_input('Merchant City')
    Merchant_State = st.text_input('Merchant_State')
    Zip = st.text_input('Zip')
    MCC	= st.text_input("MCC")
    Errors = st.selectbox('Error', ['Technical Glitch' ,'Insufficient Balance', 'Bad PIN'
    'Bad PIN,Insufficient Balance' ,'Bad PIN,Technical Glitch' ,'Bad Zipcode',
    'Insufficient Balance,Technical Glitch',
    'Bad Zipcode,Insufficient Balance', 'Bad Zipcode,Technical Glitch',
    'Bad CVV', 'Bad Expiration' ,'Bad Card Number',
    'Bad Card Number,Insufficient Balance'])
    Hour = st.selectbox('Hour', [i for i in range(24)])
    Minute = st.selectbox('Minute', [i for i in range(1,60)])

    input_data = {
        'Card': Card,
        'Year': Year,
        'Month': Month,
        'Day': Day,
        'Amount': Amount,
        'Use Chip': Use_Chip,
        'Merchant City': Merchant_City,
        'Merchant State': Merchant_State,
        'Zip': Zip,
        'MCC': MCC,
        'Errors?': Errors,
        'Hour': Hour,
        'Minute': Minute
    }
    if st.button('Enter'):
        # Preprocess the input data
        preprocessed_data = preprocess_data(input_data)
        # Load the model
        model_path = "path/to/xgboost_model.model"
        model = load_model(model_path)
        # Make predictions
        prediction = model.predict(preprocessed_data)
        st.write("Prediction:", prediction)
        on_button_click()
        st.balloons(prediction)
    
