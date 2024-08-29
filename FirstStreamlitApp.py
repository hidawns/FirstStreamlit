import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, KFold, GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = joblib.load('XGB_model.pkl')

st.title("Estimasi Tarif Konsultasi Dokter")

Experience = st.number_input('EXPERIENCE (Berapa tahun dokter tersebut berpengalaman di bidangnya)', min_value=0, max_value=66)
Rating = st.number_input('RATING (Rating Dokter)', min_value=0, max_value=100)
Profile = st.selectbox('PROFILE (Spesialisasi Dokter)', ('Dentist', 'General Medicine', 'Dermatologists', 'Homeopath', 'Ayurveda', 'ENT Specialist'))
Miscellaneous_Info = st.selectbox('MISCELLANEOUS_INFO (Deskripsi Profil)', ('Ada', 'Tidak Ada'))
#Num_of_Qualifications = 
District = st.text_input('DISTRICT')
City = st.selectbox('CITY', ('Bangalore', 'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam', 'Hyderabad', 'Mumbai', 'Thiruvananthapuram', 'Lainnya'))

