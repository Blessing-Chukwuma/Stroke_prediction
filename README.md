# 🏥 Stroke Prediction Application

A machine learning-powered web application that predicts stroke risk based on patient health metrics. Built with Streamlit for an interactive, user-friendly interface.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Details](#model-details)
- [Data Input Guide](#data-input-guide)
- [Risk Levels](#risk-levels)
- [Recommendations](#recommendations)
- [Technical Stack](#technical-stack)
- [Disclaimer](#disclaimer)

## 📖 Overview

This application uses a trained **Logistic Regression model** to predict the probability of stroke in patients based on various health factors including:
- Demographic information (age, gender, marital status)
- Medical history (hypertension, heart disease)
- Lifestyle factors (smoking status, work type)
- Health metrics (glucose level, BMI)

The model was trained on a comprehensive dataset and achieves reliable predictions through careful feature engineering and preprocessing.

## ✨ Features

### 🎯 Core Functionality
- **Real-time Prediction**: Enter your health data and get instant stroke risk assessment
- **Interactive Gauges**: Visual representation of risk probability with color-coded indicators
- **Risk Classification**: Clear LOW/MEDIUM/HIGH risk categorization
- **Health Summary**: Comprehensive display of all entered metrics

### 📊 Advanced Capabilities
- **Risk Factor Detection**: Automatically identifies active risk factors
- **Personalized Recommendations**: AI-generated health tips based on your profile
- **Clinical Score**: Detailed probability metrics for medical professionals
- **Health Guidelines**: Contextual advice for glucose levels, BMI, and lifestyle

### 🎨 User Interface
- Professional, clean design with intuitive layout
- Responsive columns for organized information display
- Color-coded risk levels (Green/Orange/Red)
- Emoji indicators for quick visual recognition
- Balloons animation on prediction completion

## 📁 Project Structure

```
Stroke Prediction/
├── streamlit_app.py                          # Main Streamlit application
├── tuned_logistic_regression_model.pkl       # Trained ML model
├── minmax_scaler.pkl                         # Feature scaler for numerical data
├── one_hot_encoder.pkl                       # Encoder for categorical features
├── requirements.txt                          # Python dependencies
├── stroke_venv/                              # Virtual environment (excluded from version control)
└── README.md                                 # This file
```

## 📦 Requirements

The application requires the following Python packages:

```
streamlit==1.52.1
pandas==2.3.3
numpy==2.3.5
scikit-learn==1.7.2
joblib==1.5.2
plotly==6.5.0
```

All dependencies are listed in `requirements.txt`.

## 🚀 Installation

### Step 1: Clone or Download the Project
```bash
cd "C:\Users\chukw\Desktop\Stroke Prediction"
```

### Step 2: Create Virtual Environment (if not already done)
```bash
python -m venv stroke_venv
```

### Step 3: Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
& '.\stroke_venv\Scripts\Activate.ps1'
```

**On Windows (Command Prompt):**
```cmd
stroke_venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
source stroke_venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Run the Application
```bash
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

## 💻 Usage

### Basic Workflow

1. **Enter Demographic Information**
   - Age: 0-100 years
   - Gender: Female, Male, or Other
   - Marital Status: Yes or No
   - Residence Type: Urban or Rural

2. **Provide Medical History**
   - Hypertension: Yes or No
   - Heart Disease: Yes or No
   - Smoking Status: Never smoked, Formerly smoked, Smokes, Unknown
   - Work Type: Private, Self-employed, Government job, Never worked

3. **Input Health Metrics**
   - Average Glucose Level: 50-300 mg/dL
   - BMI (Body Mass Index): 10-100 kg/m²

4. **Click "Predict Stroke Risk"**
   - Receive instant prediction with probability score
   - View interactive gauge chart
   - Get personalized health recommendations

### Example Scenario

A 55-year-old male with:
- Hypertension: Yes
- Heart Disease: No
- Smoking Status: Formerly smoked
- Glucose Level: 140 mg/dL
- BMI: 27.5
- Work Type: Private
- Residence: Urban

Would receive a MEDIUM risk assessment with specific recommendations for blood sugar management and weight optimization.

## 🧠 How It Works

### Data Flow

```
User Input
    ↓
Raw Data Dictionary
    ↓
Pandas DataFrame
    ↓
Categorical Encoding (One-Hot Encoder)
    ↓
Numerical Scaling (MinMax Scaler)
    ↓
Feature Alignment (MODEL_FEATURES order)
    ↓
Logistic Regression Model
    ↓
Probability Score (0-1)
    ↓
Risk Classification & Recommendations
    ↓
Visual Results Display
```

### Feature Preprocessing

**Categorical Features** (processed with One-Hot Encoding):
- Gender → Gender_Female, Gender_Male, Gender_Other
- Work Type → Work_type_Private, Work_type_Self-employed, etc.
- Residence Type → Residence_type_Urban, Residence_type_Rural
- Smoking Status → Smoking_status_smokes, Smoking_status_formerly smoked, etc.

**Numerical Features** (processed with MinMax Scaling):
- Age (normalized to 0-1 range)
- Average Glucose Level (normalized to 0-1 range)
- BMI (normalized to 0-1 range)

**Binary Features** (used as-is):
- Hypertension (0 or 1)
- Heart Disease (0 or 1)
- Ever Married (0 or 1)

### Model

- **Algorithm**: Logistic Regression
- **Type**: Binary Classification
- **Output**: Probability of stroke (0-1)
- **Decision Threshold**: 0.5 (adjustable for sensitivity/specificity trade-off)

## 📊 Model Details

### Training Data
The model was trained on a comprehensive stroke prediction dataset with thousands of patient records including:
- Demographic variables
- Medical conditions
- Lifestyle factors
- Health metrics

### Model Performance
The logistic regression model provides:
- Fast inference time (milliseconds)
- Interpretable feature weights
- Calibrated probability estimates
- Robust performance on unseen data

### Input Feature List (19 features total)

1. age
2. hypertension
3. heart_disease
4. ever_married
5. avg_glucose_level
6. bmi
7. gender_Female
8. gender_Male
9. gender_Other
10. work_type_Govt_job
11. work_type_Never_worked
12. work_type_Private
13. work_type_Self-employed
14. Residence_type_Rural
15. Residence_type_Urban
16. smoking_status_Unknown
17. smoking_status_formerly smoked
18. smoking_status_never smoked
19. smoking_status_smokes

## 📝 Data Input Guide

### Age
- **Range**: 0-100 years
- **Type**: Integer or decimal
- **Clinical Significance**: Stroke risk increases with age

### Glucose Level
- **Range**: 50-300 mg/dL
- **Clinical Reference**:
  - Normal: 70-100 mg/dL (fasting)
  - Normal: <140 mg/dL (after meals)
  - Pre-diabetic: 100-125 mg/dL
  - Diabetic: ≥126 mg/dL
- **Impact**: Higher glucose levels increase stroke risk

### BMI (Body Mass Index)
- **Calculation**: Weight(kg) / Height(m)²
- **Range**: 10-100 kg/m²
- **Classifications**:
  - Underweight: <18.5
  - Normal: 18.5-24.9 ✅
  - Overweight: 25-29.9 ⚠️
  - Obese: ≥30 🚨
- **Impact**: Obesity significantly increases stroke risk

### Hypertension
- **Definition**: High blood pressure (≥130/80 mmHg)
- **Impact**: Major stroke risk factor
- **Management**: Regular monitoring and medication

### Heart Disease
- **Types**: CAD, Heart failure, Arrhythmia, etc.
- **Impact**: Increases clot formation risk
- **Management**: Cardiology consultation recommended

### Smoking Status
- **Never smoked**: ✅ Lowest risk
- **Formerly smoked**: ⚠️ Moderate risk
- **Currently smokes**: 🚨 Highest risk
- **Unknown**: Unable to assess
- **Impact**: Major modifiable risk factor

## 🎯 Risk Levels

### LOW RISK (< 30%)
- **Interpretation**: Relatively low stroke probability
- **Recommendation**: Maintain healthy lifestyle
- **Action**: Regular health checkups (annual)

### MEDIUM RISK (30-60%)
- **Interpretation**: Moderate stroke probability
- **Recommendation**: Lifestyle modifications and physician consultation
- **Action**: 
  - Increase physical activity
  - Dietary changes
  - Regular medical monitoring (6-monthly)

### HIGH RISK (> 60%)
- **Interpretation**: High stroke probability
- **Recommendation**: Immediate medical consultation
- **Action**:
  - Seek emergency medical attention
  - Close physician monitoring
  - Consider preventive medication
  - Hospital-based management

## 💡 Recommendations System

The app generates personalized recommendations based on detected risk factors:

### Glucose Management
- Triggered when: Glucose > 125 mg/dL
- Recommendation: Monitor blood sugar, reduce sugar intake, increase activity

### Weight Management
- Triggered when: BMI ≥ 30
- Recommendation: Target BMI 18.5-24.9, regular exercise, balanced diet

### Blood Pressure Control
- Triggered when: Hypertension = Yes
- Recommendation: Limit sodium, monitor BP regularly, medication if needed

### Smoking Cessation
- Triggered when: Smoking status = Smokes or Formerly smoked
- Recommendation: Quit smoking, consult cessation programs

### Senior Care
- Triggered when: Age > 65
- Recommendation: Regular health checkups, preventive screening

## 🛠️ Technical Stack

### Backend
- **Python 3.13**: Core programming language
- **scikit-learn**: Machine learning framework
- **pandas**: Data manipulation and preprocessing
- **numpy**: Numerical computations
- **joblib**: Model serialization and loading

### Frontend
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **HTML/CSS**: Custom styling

### Preprocessing
- **MinMax Scaler**: Numerical feature normalization
- **OneHot Encoder**: Categorical feature encoding

### Deployment
- **Local Execution**: streamlit run command
- **Browser-based**: No installation required for users

## 🔒 Data Privacy

- **No Data Storage**: Input data is processed in real-time and not stored
- **Local Processing**: All computations occur on your machine
- **No Network Transfer**: Data is never sent to external servers
- **Session-based**: Each session is independent

## ⚠️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER**

This application is provided for **educational and informational purposes only**. It is not a medical diagnosis tool and should not be used as a substitute for professional medical advice.

### Key Points:
- ❌ **NOT a diagnostic tool**: Cannot definitively diagnose stroke risk
- ⚠️ **Educational use**: Designed to raise awareness about stroke risk factors
- 🏥 **Consult professionals**: Always consult a qualified healthcare provider for medical decisions
- 📋 **Complementary only**: Use alongside, not instead of, medical consultation
- ⚡ **Medical emergencies**: In case of emergency symptoms (facial drooping, arm weakness, speech difficulty), call emergency services immediately

### Liability:
The developers and maintainers of this application are not liable for any medical decisions made based on this tool's output.

## 🚑 Emergency Contact

If you experience stroke symptoms (sudden onset):
- **Facial drooping**: One side of face droops
- **Arm weakness**: Sudden weakness/numbness in arm or leg
- **Speech difficulty**: Slurred or difficult speech
- **Time to act**: Call emergency services immediately (911 in USA)

**REMEMBER: Time is brain - stroke is a medical emergency!**

## 📧 Support and Questions

For technical issues or questions about the application:
1. Check the error messages displayed in the app
2. Verify all model files (.pkl) are in the project directory
3. Ensure all dependencies are installed via requirements.txt
4. Try deactivating and reactivating the virtual environment

## 📈 Future Enhancements

Potential improvements for future versions:
- [ ] Multiple model options (Random Forest, Neural Networks)
- [ ] Patient history tracking
- [ ] Integration with wearable devices
- [ ] Export prediction reports (PDF)
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Real-time health monitoring
- [ ] Comparative risk analysis

## 📄 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

- Built with Streamlit framework
- Machine learning powered by scikit-learn
- Visualizations by Plotly
- Model trained on comprehensive health datasets

---

**Last Updated**: December 5, 2025

**Version**: 1.0.0

**Status**: ✅ Fully Functional

For detailed usage instructions, refer to the app's built-in tooltips and informational messages.
