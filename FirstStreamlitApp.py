import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost
from xgboost import XGBRegressor

model = joblib.load('XGB_model.pkl')

st.title("Hello, World!")
Profile = st.selectbox('Profile', ('Dentist', 'General Medicine', 'Dermatologists', 'Homeopath', 'Ayurveda', 'ENT Specialist'))
