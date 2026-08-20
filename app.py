import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Dropout Risk Analyzer",
    page_icon="🎓",
    layout="wide"
)

# --- 1. Model Training & Pipeline Setup (Cached) ---
@st.cache_resource(show_spinner=False)
def initialize_model():
    # Synthetic training dataset with academic and behavioral indicators
    training_data = {
        "attendance_rate": [
            95.0, 92.0, 88.0, 85.0, 81.0, 78.0, 75.0, 82.0,
            45.0, 50.0, 35.0, 52.0, 40.0, 30.0, 55.0, 42.0
        ],
        "assignment_completion": [
            90.0, 95.0, 85.0, 80.0, 75.0, 70.0, 80.0, 85.0,
            30.0, 40.0, 25.0, 45.0, 35.0, 20.0, 50.0, 38.0
        ],
        "midterm_grade": [
            88.0, 91.0, 82.0, 75.0, 79.0, 68.0, 72.0, 80.0,
            40.0, 45.0, 38.0, 42.0, 36.0, 30.0, 48.0, 41.0
        ],
        "quiz_average": [
            92.0, 94.0, 86.0, 78.0, 80.0, 72.0, 75.0, 84.0,
            35.0, 48.0, 30.0, 40.0, 33.0, 28.0, 45.0, 39.0
        ],
        "lms_logins_per_week": [
            14, 16, 12, 10, 9, 8, 11, 13,
            2, 3, 1, 3, 2, 0, 4, 2
        ],
        "late_submissions_count": [
            0, 0, 0, 1, 1, 1, 0, 0,
            5, 4, 6, 4, 5, 7, 3, 5
        ],
        "label": [
            0, 0, 0, 0, 0, 0, 0, 0,  # Safe students
            1, 1, 1, 1, 1, 1, 1, 1   # At-risk students
        ]
    }
    df = pd.DataFrame(training_data)
    X = df.drop(columns=["label"])
    y = df["label"]

    # Random Forest handles small datasets cleanly without binning constraints
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y)

    # Initialize Tree SHAP explainer
    explainer = shap.TreeExplainer(model)
    return model, explainer, list(X.columns)

model, explainer, feature_names = initialize_model()

# --- 2. User Interface ---
st.title("🎓 Student Dropout Risk Prediction & Early Warning System")
st.markdown("Enter student academic performance and engagement metrics to estimate dropout probability and review primary risk drivers.")

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Student Metrics Input")
    student_name = st.text_input("Student Identifier / Name", value="Jane Doe")
    
    attendance = st.slider("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=65.0, step=0.5)
    assignment = st.slider("Assignment Completion (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
    midterm = st.slider("Midterm Exam Grade", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
    quiz = st.slider("Quiz Average Score", min_value=0.0, max_value=100.0, value=58.0, step=1.0)
    logins = st.number_input("LMS Logins Per Week", min_value=0, max_value=50, value=6, step=1)
    late_submissions = st.number_input("Late Submissions Count", min_value=0, max_value=20, value=2, step=1)

# --- 3. Inference and SHAP Calculations ---
input_dict = {
    "attendance_rate": attendance,
    "assignment_completion": assignment,
    "midterm_grade": midterm,
    "quiz_average": quiz,
    "lms_logins_per_week": logins,
    "late_submissions_count": late_submissions
}
input_df = pd.DataFrame([input_dict])[feature_names]

# Generate Random Forest prediction
prob = float(model.predict_proba(input_df)[0][1])
is_at_risk = prob >= 0.50

# Calculate local feature attribution via SHAP
shap_raw = explainer.shap_values(input_df)

# Handle output structure across SHAP versions for Random Forest
if isinstance(shap_raw, list):
    shap_vals = shap_raw[1][0]
elif len(np.shape(shap_raw)) == 3:
    shap_vals = shap_raw[0, :, 1]
else:
    shap_vals = shap_raw[0]

with col2:
    st.subheader(f"Risk Assessment for {student_name}")
    
    # Visual Metric Indicators
    m1, m2, m3 = st.columns(3)
    
    with m1:
        if prob >= 0.70:
            st.error("High Risk Tier")
        elif prob >= 0.40:
            st.warning("Medium Risk Tier")
        else:
            st.success("Low Risk Tier")
            
    with m2:
        st.metric(label="Dropout Probability", value=f"{prob * 100:.1f}%")
        
    with m3:
        st.metric(label="Binary Status", value="Flagged" if is_at_risk else "Normal")

    st.markdown("---")
    st.subheader("Intervention Drivers (SHAP Explainability)")
    st.caption("Red bars push the student toward dropping out; blue bars indicate protective factors.")

    # Render Bar Chart for Feature Impacts
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y_pos = np.arange(len(feature_names))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_vals]

    ax.barh(y_pos, shap_vals, align="center", color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace("_", " ").title() for f in feature_names])
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Impact on Dropout Risk (SHAP Value)")
    
    plt.tight_layout()
    st.pyplot(fig)