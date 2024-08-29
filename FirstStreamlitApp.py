import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# Load the trained model
model = joblib.load('XGB_model (1).pkl')

# Function to preprocess user inputs
def preprocess_input(experience, num_of_qualifications, rating, miscellaneous_info, profile, place):
    # Create a DataFrame with user inputs
    input_df = pd.DataFrame({
        'Experience': [np.sqrt(experience)],
        'Num_of_Qualifications': [num_of_qualifications],
        'Rating': [rating],
        'Miscellaneous_Info': [1 if miscellaneous_info == 'Present' else 0],
        'Profile': [profile],
        'Place': [place]
    })

    # Encode the Profile and Place columns
    le_profile = LabelEncoder()
    input_df['Profile'] = le_profile.fit_transform(input_df['Profile'])

    le_place = LabelEncoder()
    input_df['Place'] = le_place.fit_transform(input_df['Place'])

    # Scaling
    scaler = MinMaxScaler()
    cols_to_scale = ['Experience', 'Num_of_Qualifications', 'Rating', 'Miscellaneous_Info', 'Profile', 'Place']
    input_df[cols_to_scale] = scaler.fit_transform(input_df[cols_to_scale])

    return input_df

# Streamlit app layout
def main():
    st.title("Doctor Salary Prediction")

    # Input fields for user to provide data for prediction
    experience = st.number_input('Years of Experience', min_value=0, max_value=66, step=1)
    num_of_qualifications = st.number_input('Number of Qualifications', min_value=1, max_value=10, step=1)
    rating = st.number_input('Doctor Rating', min_value=1, max_value=100, step=1)
    
    miscellaneous_info = st.selectbox('Miscellaneous Info Existent', ['Not Present', 'Present'])
    
    profile = st.selectbox('Doctor Specialization', ['Ayurveda', 'Dentist', 'Dermatologist', 'ENT Specialist', 'General Medicine', 'Homeopath'])
    
    place = st.selectbox('Place', ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 'Coimbatore', 'Ernakulam', 'Thiruvananthapuram', 'Other'])

    # Prediction button
    if st.button('Predict'):
        input_data = preprocess_input(experience, num_of_qualifications, rating, miscellaneous_info, profile, place)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        st.write(f"Predicted Doctor Salary: ₹{round(prediction[0], 2)}")

if __name__ == '__main__':
    main()
