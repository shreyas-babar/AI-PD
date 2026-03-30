import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("Student Depression Prediction")

# Load dataset
df = pd.read_csv("student_depression_datasets.csv")
df = df.drop("id", axis=1)

# One-hot encoding (IMPORTANT)
df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop("Depression", axis=1)
y = df["Depression"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# =====================
# INPUT SECTION
# =====================

st.write("Enter Details:")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 100)

academic_pressure = st.slider("Academic Pressure (1-10)", 1, 10)
study_hours = st.slider("Study Hours (1-10)", 1, 10)
cgpa = st.slider("CGPA (1-10)", 1, 10)
financial_stress = st.slider("Financial Stress (1-10)", 1, 10)
diet = st.slider("Diet Habits (1-10)", 1, 10)

# =====================
# CREATE INPUT DATAFRAME (MATCH TRAINING FORMAT)
# =====================

input_dict = {
    "Age": age,
    "Academic Pressure": academic_pressure,
    "Study Hours": study_hours,
    "CGPA": cgpa,
    "Financial Stress": financial_stress,
    "Diet Habits": diet,
    "Gender_Male": 1 if gender == "Male" else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Align columns with training data
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# =====================
# PREDICTION
# =====================

if st.button("Predict"):
    pred = model.predict(input_df)

    if pred[0] == 1:
        st.error("Depressed")
    else:
        st.success("Not Depressed")