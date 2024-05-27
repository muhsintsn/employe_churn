import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
st.title("Churn PAGE")
randomf_model = open("RandomForestClassifier.pkl","rb")
randomf_model = joblib.load(randomf_model)



st.sidebar.title('Configure Your Customer')
satisfaction_level = st.sidebar.number_input("satisfaction_level", value=0.10)
last_evaluation = st.sidebar.number_input("last_evaluation", value=0.93)
number_project = st.sidebar.number_input("number_project", value=6)
average_montly_hours = st.sidebar.number_input("average_montly_hours",  value=270)
time_spend_company = st.sidebar.number_input("time_spend_company", value=4)


data = {}
data["satisfaction_level"]=satisfaction_level
data["last_evaluation"]=last_evaluation
data["number_project"]=number_project
data["average_montly_hours"]=average_montly_hours
data["time_spend_company"]=time_spend_company
predict = st.sidebar.button("P R E D I C T")


if predict:
    df = pd.DataFrame([data])
    result = randomf_model.predict(df)
    st.table(pd.DataFrame([data]))
    st.write(result)
    if result == 0:
        st.markdown("<h2 style='text-align: center; color: green;'>He/she will stay.</h2>", unsafe_allow_html=True)
    elif result == 1:
        st.markdown("<h2 style='text-align: center; color: red;'>He/she will not stay.</h2>", unsafe_allow_html=True)


