
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model_path = Path(__file__).parent / "student_pass_fail_model.pkl"

if not model_path.exists():
    st.error("Model file not found!")
    st.stop()

model = joblib.load(model_path)

st.title("Student Pass Predictor")
st.write("Enter the student's study hours and attendance.")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Study Hours": [study_hours],
        "Attendence": [attendance]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][int(prediction)]

    if prediction == 1:
        st.success(
            f"Predicted Result: PASS ({probability:.1%} confidence)"
        )
    else:
        st.error(
            f"Predicted Result: FAIL ({probability:.1%} confidence)"
        )
