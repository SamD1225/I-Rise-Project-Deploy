import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="HealthCheck Predictor", page_icon="🩺", layout="wide")

# Custom CSS for centering and styling
st.markdown("""
    <style>
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }
    h1, h3, p {
        text-align: center;
    }
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
        background-color: #4CAF50;
        color: white;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Header
st.title("🩺 Diabetes Risk Assessment")
st.write("Complete the health profile below to see your prediction.")

# Centering using columns
empty_l, col, empty_r = st.columns([1, 3, 1])

with col:
    with st.container():
        # 1. Pregnancies (Categorical Selection)
        preg_label = st.select_slider(
            "Number of Pregnancies",
            options=["None", "1-2", "3-5", "6-9", "10+"]
        )
        # Map labels back to numeric averages/values for the model
        preg_map = {"None": 0, "1-2": 2, "3-5": 4, "6-9": 7, "10+": 12}
        pregnancies = preg_map[preg_label]

        # 2. Glucose (Categorical)
        gluc_label = st.selectbox(
            "Glucose Level",
            options=["Normal (Under 100)", "Prediabetes (100-125)", "High (126+)"]
        )
        gluc_map = {"Normal (Under 100)": 90, "Prediabetes (100-125)": 115, "High (126+)": 150}
        glucose = gluc_map[gluc_label]

        # 3. Blood Pressure (Categorical)
        bp_label = st.selectbox(
            "Blood Pressure Range",
            options=["Low (Under 60)", "Normal (60-80)", "High (80-90)", "Very High (90+)"]
        )
        bp_map = {"Low (Under 60)": 50, "Normal (60-80)": 70, "High (80-90)": 85, "Very High (90+)": 100}
        blood_pressure = bp_map[bp_label]

        # Remaining inputs using sliders for a better UI than text boxes
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)
        insulin = st.slider("Insulin Level (mu U/ml)", 0, 846, 79)
        bmi = st.slider("BMI (Body Mass Index)", 0.0, 67.1, 32.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.4, 0.5)
        age = st.slider("Age", 21, 81, 33)

    st.markdown("---")
    
    # Prediction Button
    if st.button("Generate Prediction"):
        # Arrange features for the model
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, dpf, age]])
        
        prediction = model.predict(features)
        
        # Centered Results Display
        if prediction[0] == 1:
            st.error("### ⚠️ High Risk Detected")
            st.write("Based on the data provided, the model suggests a higher likelihood of diabetes. Please consult a doctor for a formal diagnosis.")
        else:
            st.success("### ✅ Low Risk Detected")
            st.write("The model suggests a low likelihood of diabetes. Maintain your healthy habits!")
