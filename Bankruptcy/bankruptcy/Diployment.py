import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px


# Load the trained Random Forest model and scaler
with open('ruptcy_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Sidebar Configuration
with st.sidebar.header("🗂️ Navigation"):
    sidebar_options = [ "🧑‍💻 Prediction","💾 About"]
    selected_option = st.sidebar.radio("Select a Section", sidebar_options)

# Streamlit app title
st.title("Bankruptcy Prediction")

# Input fields for the features
st.header("Enter the business risk parameters:")

def selection(label):
    options=["select option","Low Risk","Medium Risk","High Risk"]

    selected_option= st.selectbox(label=label,options=options)
    if selected_option=="Low Risk":
        return 0
    if selected_option=="Medium Risk":
        return 0.5
    if selected_option== "High Risk":
        return 1
    
if selected_option=='🧑‍💻 Prediction':


    industrial_risk = selection("Indestrial Risk")
    management_risk =selection("Management Risk")
    financial_flexibility =selection("Financial Flexibility")
    credibility = selection("Credibility")
    competitiveness = selection('Competitiveness')
    operating_risk = selection("Operating Risk")

    # Predict button
    if st.button("Predict Bankruptcy Risk"):
        try:
            # Prepare the input features
            features = np.array([
                industrial_risk,
                management_risk,
                financial_flexibility,
                credibility,
                competitiveness,
                operating_risk
            ]).reshape(1, -1)

        

            # Make a prediction
            prediction = model.predict(features)
            #prediction_proba = model.predict_proba(features_scaled)

            # Display the result
            if prediction == 1:
                st.success("The model predicts **Non Bankruptcy Risk**.")
            else:
                st.success("The model predicts **Bankruptcy Risk**.")
            
        

        except Exception as e:
            st.error(f"Error: {str(e)}")

#Loading Data set
data=pd.read_excel(r"bankruptcy.xlsx")
#Loading metrics df
metrics=pd.read_excel(r"metrics.xlsx")


if selected_option=="💾 About":
    
    #About Section
    st.title(":green[About]")
    st.markdown(
        """
        ### Bank Ruptcy Dashboard
        This app provides **Bank Ruptcy predication** based on bankruptcy data using **Decision Tree Regressor**.

        ### Model Description:
        - A Decision Tree Regressor is utilized to build the model.
        - A XGBoost model is used incorporates feature importance of the columns.
        - The Decision Tree model is performed well and given the more accuracy.

        ### Key Features:
        - Uploading low, medium, high risk information to get the predictions.
        - geting the Rupctcy and Non-Rupctcy as outputs.
        - Examine data set using train the model.
        
        ### Built model with:
        - Python
        - streamlit
        - Machine learning 
        """
                
    )

    options = ['Select Options', 'data', 'Visualization', 'metrics', 'Feature importance']
    #dropdown box
    selected_data = st.selectbox("Click Here For More:", options)
    #Display The Selected Optins
    if selected_data == 'data':
        #Display Data set
        st.title(" :green[Dataset]")
        st.markdown('''
        ### Data set Overview
        - Industrial risk: It contains three variables Low Risk=0, Medium=0.5 and High=1 Risk.
        - Management risk: It contains three variables Low Risk=0, Medium=0.5 and High=1 Risk.
        - Financial Flexibility: It contains three variables Low Risk=0, Medium=0.5 and High=1 Risk.
        - Credibility: It contains three variables Low Risk=0, Medium=0.5 and High=1 Risk.
        - Competitiveness: It contains three variables Low Risk=0, Medium=0.5 and High=1 Risk.
        - Operating Risk: It contains three variables Low Risk=0, Medium=0.5 and High=1 Risk.
        - Class : It contains three variables Low Risk=0, Medium=0.5 and High=1 Risk.                                                      
                    ''')
        st.dataframe(data, use_container_width=True)
    if selected_data == 'Visualization':
        # Display plots
        st.title(" :orange[Interactive Histograms]")
        for column in data.columns:
            fig = px.histogram(data, x=column, title=f"{column} Distribution")
            st.plotly_chart(fig)

    if selected_data == "metrics":
        #Display Metrix
        st.title(" :green [metrics of different models]")
        st.dataframe(metrics)        
    if selected_data == 'Feature importance':
        #Display importance
        st.title(" :green Feature importance")
        st.markdown('''
                
                - industrial_risk: 0.00
                - management_risk: 0.00
                - financial_flexibility: 0.00
                - credibility: 0.07
                - competitiveness: 0.93
                - operating_risk: 0.00
                ''')    
    