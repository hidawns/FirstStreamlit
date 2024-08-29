import streamlit as st

model = joblib.load('XGB_model.pkl')

st.title("Hello, World!")
Profile = st.selectbox('Profile', ('Dentist', 'General Medicine', 'Dermatologists', 'Homeopath', 'Ayurveda', 'ENT Specialist'))
