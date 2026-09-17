```python
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model = joblib.load(
    Path(__file__).parent / "student_pass_fail_model.pkl"
)

st.title("Student Pass Predictor")
st.write("Enter the student's study hours and attendance.")

hours = st.number_input("Study Hours", 0.0, 24.0, 5.0)
attendance = st.number_input("Attendance (%)", 0.0, 100.0, 75.0)

if st.button("Predict"):

    feature = model.feature_names_in_[0]

    data = pd.DataFrame({
        feature: [hours]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][int(prediction)]

    if prediction == 1:
        st.success(f"Predicted Result: PASS ({probability:.1%} confidence)")
    else:
        st.error(f"Predicted Result: FAIL ({probability:.1%} confidence)")

    st.info(f"Attendance entered: {attendance:.0f}%")
```
