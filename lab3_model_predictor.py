import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Lab 3 - Model Prediction", layout="wide")
st.title("ITD105 – Lab 3: Model Prediction (Part 3)")

menu = ["Classification: Heart Disease", "Regression: Air Temperature"]
choice = st.sidebar.radio("Choose Model Type", menu)

model_file = st.file_uploader("Upload Trained Model (.joblib)", type="joblib")

if model_file:
    model = joblib.load(model_file)
    st.success("Model loaded successfully.")

    if choice == "Classification: Heart Disease":
        st.header("Heart Disease Prediction")

        with st.form("classification_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", value=50)
                sex = st.selectbox("Sex", ["Male", "Female"])
                cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
                trestbps = st.number_input("Resting Blood Pressure", value=120)
                chol = st.number_input("Cholesterol", value=200)
                fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
                restecg = st.selectbox("Resting ECG", [0, 1, 2])
            with col2:
                thalach = st.number_input("Max Heart Rate", value=150)
                exang = st.selectbox("Exercise Induced Angina", [0, 1])
                oldpeak = st.number_input("Oldpeak", value=1.0)
                slope = st.selectbox("Slope", [0, 1, 2])
                thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
            submitted = st.form_submit_button("Predict")

        if submitted:
            sex_val = 1 if sex == "Male" else 0
            features = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg,
                                  thalach, exang, oldpeak, slope, thal]])

            expected_features = model.n_features_in_
            if features.shape[1] != expected_features:
                st.error(f"Model expects {expected_features} features, but received {features.shape[1]}.")
            else:
                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0][1] * 100
                label = "Has Heart Disease" if pred == 1 else "No Heart Disease"
                st.success(f"Prediction: {label}")
                st.info(f"Probability: {prob:.2f}%")

    elif choice == "Regression: Air Temperature":
        st.header("Air Temperature Prediction")

        feature_names = [
            "Humidity", "Wind Speed", "Pressure", "Visibility",
            "Cloud Cover", "Precipitation", "Dew Point", "UV Index",
            "Ozone", "Solar Radiation", "Wind Direction", "Sea Level Pressure"
        ]

        default_values = [60.0, 5.5, 1012.0, 10.0, 40.0, 0.0, 12.5, 7.0, 300.0, 500.0, 180.0, 1015.0]

        if not hasattr(model, "n_features_in_"):
            st.error("Uploaded model does not contain feature metadata.")
        else:
            expected_features = model.n_features_in_
            st.info(f"This model expects {expected_features} input features.")

            with st.form("regression_form"):
                inputs = []
                col1, col2 = st.columns(2)
                for i in range(expected_features):
                    label = feature_names[i] if i < len(feature_names) else f"Feature {i+1}"
                    default = default_values[i] if i < len(default_values) else 0.0
                    with (col1 if i % 2 == 0 else col2):
                        val = st.number_input(label, value=default, key=f"reg_input_{i}")
                        inputs.append(val)
                submitted = st.form_submit_button("Predict")

            if submitted:
                features = np.array([inputs])
                if features.shape[1] != expected_features:
                    st.error(f"Model expects {expected_features} features, but received {features.shape[1]}.")
                else:
                    pred = model.predict(features)[0]
                    st.success(f"Predicted Air Temperature for Tomorrow: {pred:.2f}°C")
