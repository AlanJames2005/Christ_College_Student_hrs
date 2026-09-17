import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model_path = Path(__file__).parent / "student_pass_fail_model.pkl"
model = joblib.load(model_path)

st.title("Student Pass Predictor")
st.write("Enter the student's study hours and attendance.")

study_hours = st.number_input("Study Hours", 0.0, 24.0, 5.0, 0.5)
attendance = st.number_input("Attendance (%)", 0.0, 100.0, 75.0, 1.0)

if st.button("Predict"):

    # Show the names the model actually expects
    st.write("Model expects:", model.feature_names_in_)

    names = model.feature_names_in_

    input_data = pd.DataFrame({
        names[0]: [study_hours],
        names[1]: [attendance]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][int(prediction)]

    if prediction == 1:
        st.success(f"Predicted Result: PASS ({probability:.1%} confidence)")
    else:
        st.error(f"Predicted Result: FAIL ({probability:.1%} confidence)")
