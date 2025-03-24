#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster


# In[11]:


import warnings
warnings.filterwarnings('ignore')


# In[13]:


# Load pre-trained model and scaler
scaler = pickle.load(open("scaler.pkl", 'rb'))
pca = pickle.load(open("pca.pkl", 'rb'))
linkage_matrix = pickle.load(open("hierarchical_model.pkl", 'rb'))  # Precomputed linkage matrix


# In[15]:


# Feature selection 
selected_features = ['age', 'gender', 'chest_pain_type', 'blood_pressure', 'cholesterol', 
                      'max_heart_rate', 'exercise_angina', 'plasma_glucose', 'skin_thickness', 
                      'insulin', 'bmi', 'diabetes_pedigree', 'hypertension', 'heart_disease', 
                      'smoking_status_Smoker', 'smoking_status_Unknown', 'residence_type_Urban']

# Streamlit UI
st.title("Patient Triage Clustering System")
st.write("This app predicts patient priority based on symptoms using Hierarchical Clustering.")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", [0, 1])
chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=300, step=1)
cholesterol = st.number_input("Cholesterol Level", min_value=120, max_value=300, step=1)
max_heart_rate = st.number_input("Max Heart Rate", min_value=70, max_value=220, step=1)
exercise_angina = st.selectbox("Exercise Angina", [0, 1])
plasma_glucose = st.number_input("Plasma Glucose", min_value=50, max_value=300, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_status = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker", "Unknown"])
residence = st.selectbox("Residence Type", ["Rural", "Urban"])

# Convert smoking status to one-hot encoding
smoking_status_Smoker = 1 if smoking_status == "Smoker" else 0
smoking_status_Unknown = 1 if smoking_status == "Unknown" else 0
residence_type_Urban = 1 if residence == "Urban" else 0

if st.button("Predict Patient Priority"):
    # Create DataFrame with user input
    input_data = pd.DataFrame([[age, gender, chest_pain, blood_pressure, cholesterol, max_heart_rate, 
                                exercise_angina, plasma_glucose, skin_thickness, insulin, bmi, 
                                diabetes_pedigree, hypertension, heart_disease, smoking_status_Smoker, 
                                smoking_status_Unknown, residence_type_Urban]],
                              columns=selected_features)

    # Preprocessing
    scaled_data = scaler.transform(input_data)  # Scale the input data
    pca_data = pca.transform(scaled_data)[:, :2]  # Apply PCA with 2 components

    # Load the original PCA-transformed dataset (used during training)
    original_pca_data = pickle.load(open("pca_transformed.pkl", "rb"))  # Load stored PCA-transformed data

    # Append new patient data to the original PCA-transformed dataset
    combined_pca_data = np.vstack([original_pca_data, pca_data])  # Add new patient data

    # Compute hierarchical clustering dynamically
    new_linkage_matrix = linkage(combined_pca_data, method='average')

    # Get cluster labels for the updated dataset
    cluster_labels = fcluster(new_linkage_matrix, 3, criterion='maxclust')

    # Assign the last processed point (patient input) to a cluster
    cluster = cluster_labels[-1] - 1  # Adjust index to start from 0

    # Display patient cluster
    st.success(f"Patient is categorized into Cluster {cluster}")

    # Define priority levels
    if cluster == 1:
        st.warning("High Priority - Requires immediate medical attention")
    elif cluster == 0:
        st.info("Medium Priority - Needs medical attention but not an emergency")
    else:
        st.success("Low Priority - Routine check-up recommended, but no urgent risk")

st.write("This system helps prioritize emergency medical care based on patient symptoms.")


# In[ ]:




