import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# Title
st.title("Telco Customer Churn Dashboard")

# Load dataset
file_path=df = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

# Preprocessing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Features and target
X = df[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']]
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Dataset toggle
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# Summary Statistics
if st.checkbox("Show Summary Statistics"):
    st.write(df.describe())

# Churn Distribution
st.subheader("Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Churn', data=df, ax=ax)
ax.set_xticklabels(["No", "Yes"])
st.pyplot(fig)

# Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df[['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','Churn']].corr(),
            cmap='coolwarm', annot=True, ax=ax)
st.pyplot(fig)

# Model Metrics
st.subheader("Model Evaluation")
st.write("Accuracy:", model.score(X_test_scaled, y_test))
st.text(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, th = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC={auc:.2f}")
ax.plot([0, 1], [0, 1], 'k--')
st.pyplot(fig)

# Prediction for new customer
st.sidebar.header("Predict Churn for New Customer")
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
tenure_val = st.sidebar.number_input("Tenure", min_value=0, max_value=100, value=12)
monthly = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0)
total_val = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

new_data = scaler.transform([[senior, tenure_val, monthly, total_val]])
pred = model.predict(new_data)[0]
proba = model.predict_proba(new_data)[0][1]

st.subheader("Prediction Result")
if pred == 1:
    st.warning(f"This customer is likely to CHURN. Probability: {proba:.2f}")
else:
    st.success(f"This customer will likely STAY. Churn probability: {proba:.2f}")

