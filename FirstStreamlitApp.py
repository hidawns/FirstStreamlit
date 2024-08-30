import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# Load the trained model
model = joblib.load('XGB_model (1).pkl')

# Function to preprocess user inputs
def preprocess_input(experience, rating, place, profile, miscellaneous_info, num_of_qualifications):
    # Create a DataFrame with user inputs

    place_mapping = {'Bangalore': 0, 'Chennai': 1, 'Coimbatore': 2, 'Delhi': 3, 'Ernakulam': 4, 'Hyderabad': 5, 'Mumbai': 6, 'Thiruvananthapuram': 7, 'Other': 8
    }
    
    # Mapping for 'Profile'
    profile_mapping = {'Ayurveda': 0, 'Dentist': 1, 'Dermatologist': 2, 'ENT Specialist': 3, 'General Medicine': 4, 'Homeopath': 5
    }
    
    input_df = pd.DataFrame({
        'Experience': [np.sqrt(experience)],
        'Rating': [rating],
        'Place': [place_mapping[place]],
        'Profile': [profile_mapping[profile]],
        'Miscellaneous_Info': [1 if miscellaneous_info == 'Present' else 0],
        'Num_of_Qualifications': [num_of_qualifications],
        'Fee_category': [0.0]
    })

    # Encode the Profile and Place columns
    #le_place = LabelEncoder()
    #input_df['Place'] = le_place.fit_transform(input_df['Place'])
    
    #le_profile = LabelEncoder()
    #input_df['Profile'] = le_profile.fit_transform(input_df['Profile'])

    # Scaling
    scaler = MinMaxScaler()
    cols_to_scale = ['Experience', 'Rating', 'Place', 'Profile', 'Miscellaneous_Info', 'Num_of_Qualifications', 'Fee_category']
    input_df[cols_to_scale] = scaler.fit_transform(input_df[cols_to_scale])

    return input_df

# Streamlit app layout
def main():
    st.title("Doctor Salary Prediction")

    # Input fields for user to provide data for prediction
    experience = st.number_input('Years of Experience', min_value=0, max_value=66, step=1)
    rating = st.number_input('Doctor Rating', min_value=1, max_value=100, step=1)
    place = st.selectbox('Place', ['Bangalore',  'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam', 'Hyderabad', 'Mumbai', 'Thiruvananthapuram', 'Other'])
    profile = st.selectbox('Doctor Specialization', ['Ayurveda', 'Dentist', 'Dermatologist', 'ENT Specialist', 'General Medicine', 'Homeopath'])
    miscellaneous_info = st.selectbox('Miscellaneous Info Existent', ['Not Present', 'Present'])
    num_of_qualifications = st.number_input('Number of Qualifications', min_value=1, max_value=10, step=1)
    
    # Prediction button
    if st.button('Predict'):
        input_data = preprocess_input(experience, rating, place, profile, miscellaneous_info, num_of_qualifications)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        st.write(f"Predicted Doctor Salary: {round(prediction[0], 2)}")

if __name__ == '__main__':
    main()
