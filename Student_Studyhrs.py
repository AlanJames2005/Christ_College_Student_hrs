import streamlit as st
import pandas as pd
import joblib

model = joblib.load("student_pass_fail_model.pkl")

st.title("Student Pass Predictor")

hours = st.number_input("Study Hours", min_value=0.0)

if st.button("Predict"):
    name = model.feature_names_in_[0]
    data = pd.DataFrame({name: [hours]})
    result = model.predict(data)[0]

    if result == 1:
        st.success("Student will Pass")
    else:
        st.error("Student will Fail")
