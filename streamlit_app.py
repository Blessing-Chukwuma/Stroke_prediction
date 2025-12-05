import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model and preprocessors
@st.cache_resource
def load_models():
    try:
        model = joblib.load('tuned_logistic_regression_model.pkl')
        scaler = joblib.load('minmax_scaler.pkl')
        one_hot_encoder = joblib.load('one_hot_encoder.pkl')
        return model, scaler, one_hot_encoder
    except FileNotFoundError:
        st.error("Model or preprocessor files not found. Make sure all .pkl files are in the same directory.")
        st.stop()

model, scaler, one_hot_encoder = load_models()

# Define the feature names in the exact order the model expects
MODEL_FEATURES = [
    'age', 'hypertension', 'heart_disease', 'ever_married',
    'avg_glucose_level', 'bmi', 'gender_Female', 'gender_Male',
    'gender_Other', 'work_type_Govt_job', 'work_type_Never_worked',
    'work_type_Private', 'work_type_Self-employed',
    'Residence_type_Rural', 'Residence_type_Urban',
    'smoking_status_Unknown', 'smoking_status_formerly smoked',
    'smoking_status_never smoked',
    'smoking_status_smokes'
]

# Page header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🏥 Stroke Risk Predictor")
    st.markdown("*Predict your risk of stroke based on health metrics*")
with col2:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

st.divider()

# Preprocessing Function
def preprocess_input(df_input, scaler, one_hot_encoder):
    categorical_cols = ['gender', 'work_type', 'Residence_type', 'smoking_status']
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']

    encoded_features = one_hot_encoder.transform(df_input[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(categorical_cols), index=df_input.index)

    df_processed = pd.concat([
        df_input[['hypertension', 'heart_disease', 'ever_married']],
        df_input[numerical_cols],
        encoded_df
    ], axis=1)

    df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])

    final_input = pd.DataFrame(0, index=df_processed.index, columns=MODEL_FEATURES)
    for col in df_processed.columns:
        if col in final_input.columns:
            final_input[col] = df_processed[col]
    
    final_input = final_input[MODEL_FEATURES]
    return final_input

# Create two columns for input layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Demographic Information")
    age = st.slider('Age (years)', min_value=0, max_value=100, value=45, step=1)
    gender = st.radio('Gender', options=['Female', 'Male', 'Other'], horizontal=True)
    ever_married = st.selectbox('Marital Status', options=['Yes', 'No'])
    residence_type = st.radio('Residence Type', options=['Urban', 'Rural'], horizontal=True)

with col2:
    st.subheader("❤️ Medical History")
    hypertension = st.selectbox('Hypertension', options=['No', 'Yes'], key='hyp')
    heart_disease = st.selectbox('Heart Disease', options=['No', 'Yes'], key='hd')
    smoking_status = st.selectbox('Smoking Status', options=['never smoked', 'formerly smoked', 'smokes', 'Unknown'])
    work_type = st.selectbox('Work Type', options=['Private', 'Self-employed', 'Govt_job', 'Never_worked'])

st.divider()

# Health metrics section
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Health Metrics")
    avg_glucose_level = st.number_input('Average Glucose Level (mg/dL)', min_value=50, max_value=300, value=120, step=5)
    st.caption("Normal: 70-100 fasting, <140 after meals")

with col2:
    bmi = st.number_input('BMI (Body Mass Index)', min_value=10.0, max_value=100.0, value=24.5, step=0.5)
    st.caption("Normal: 18.5-24.9, Overweight: 25-29.9")

with col3:
    st.subheader("ℹ️ Risk Factors")
    risk_factors = []
    if hypertension == 'Yes':
        risk_factors.append("🩸 Hypertension")
    if heart_disease == 'Yes':
        risk_factors.append("❤️ Heart Disease")
    if smoking_status in ['smokes', 'formerly smoked']:
        risk_factors.append("🚬 Smoking")
    if bmi >= 30:
        risk_factors.append("⚠️ Obesity (BMI ≥ 30)")
    if avg_glucose_level > 125:
        risk_factors.append("📈 High Glucose")
    
    if risk_factors:
        for factor in risk_factors:
            st.info(factor)
    else:
        st.success("✅ No major risk factors detected")

st.divider()

# Prepare prediction
hypertension_encoded = 1 if hypertension == 'Yes' else 0
heart_disease_encoded = 1 if heart_disease == 'Yes' else 0
ever_married_encoded = 0 if ever_married == 'Yes' else 1

raw_input_data = {
    'age': age,
    'hypertension': hypertension_encoded,
    'heart_disease': heart_disease_encoded,
    'ever_married': ever_married_encoded,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'gender': gender,
    'work_type': work_type,
    'Residence_type': residence_type,
    'smoking_status': smoking_status
}

input_df = pd.DataFrame([raw_input_data])

# Prediction button and results
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button('🔍 Predict Stroke Risk', use_container_width=True, key='predict_btn'):
        st.balloons()
        processed_input = preprocess_input(input_df.copy(), scaler, one_hot_encoder)
        prediction_proba = model.predict_proba(processed_input)[:, 1][0]
        prediction = model.predict(processed_input)[0]
        
        # Display results
        st.subheader("📈 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction_proba < 0.3:
                risk_level = "LOW"
                risk_color = "green"
                emoji = "✅"
            elif prediction_proba < 0.6:
                risk_level = "MEDIUM"
                risk_color = "orange"
                emoji = "⚠️"
            else:
                risk_level = "HIGH"
                risk_color = "red"
                emoji = "🚨"
            
            st.metric(
                label=f"{emoji} Risk Level",
                value=risk_level,
                delta=f"{prediction_proba*100:.1f}% probability"
            )
        
        with col2:
            fig = go.Figure(data=[go.Gauge(
                mode="gauge+number+delta",
                value=prediction_proba * 100,
                title={'text': "Stroke Risk (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            )])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown(f"### Clinical Score: {prediction_proba:.3f}")
            st.markdown("---")
            if prediction_proba < 0.3:
                st.success("✅ **LOW RISK**: Your stroke risk is relatively low based on current metrics.")
            elif prediction_proba < 0.6:
                st.warning("⚠️ **MEDIUM RISK**: Consider lifestyle modifications and consult your physician.")
            else:
                st.error("🚨 **HIGH RISK**: Please seek immediate medical consultation.")
        
        # Risk breakdown
        st.markdown("---")
        st.subheader("📊 Health Summary")
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.metric("Age", f"{age} years", "")
            st.metric("BMI", f"{bmi:.1f}", "")
        
        with summary_col2:
            st.metric("Glucose", f"{avg_glucose_level} mg/dL", "")
            st.metric("Hypertension", hypertension, "")
        
        with summary_col3:
            st.metric("Heart Disease", heart_disease, "")
            st.metric("Smoking", smoking_status, "")
        
        with summary_col4:
            st.metric("Gender", gender, "")
            st.metric("Work Type", work_type, "")
        
        # Health recommendations
        st.markdown("---")
        st.subheader("💡 Health Recommendations")
        
        recommendations = []
        if avg_glucose_level > 125:
            recommendations.append("🍎 **Monitor Blood Sugar**: Your glucose level is elevated. Reduce sugar intake and increase physical activity.")
        if bmi >= 30:
            recommendations.append("🏃 **Weight Management**: Aim for a BMI between 18.5-24.9. Regular exercise and balanced diet are key.")
        if hypertension == 'Yes':
            recommendations.append("🧂 **Blood Pressure Control**: Limit sodium intake and monitor blood pressure regularly.")
        if smoking_status in ['smokes', 'formerly smoked']:
            recommendations.append("🚭 **Smoking Cessation**: Quitting smoking significantly reduces stroke risk.")
        if age > 65:
            recommendations.append("👴 **Senior Care**: Regular health checkups are essential for your age group.")
        if not recommendations:
            recommendations.append("✅ **Maintain Healthy Lifestyle**: Keep up with regular exercise and balanced diet.")
        
        for rec in recommendations:
            st.info(rec)

st.divider()

# Footer
st.markdown("""
---
**Disclaimer**: This tool provides an educational prediction based on the input data. 
It should not replace professional medical advice. Always consult with a qualified healthcare provider for proper diagnosis and treatment.

**Data Privacy**: Your data is processed locally and not stored.
""")