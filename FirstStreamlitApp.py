import streamlit as st
import pandas as pd
import joblib

model = joblib.load('XGB_model.pkl')

st.title("Hello, World!")
Profile = st.selectbox('Profile', ('Dentist', 'General Medicine', 'Dermatologists', 'Homeopath', 'Ayurveda', 'ENT Specialist'))
