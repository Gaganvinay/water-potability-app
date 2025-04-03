import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("💧 Water Potability Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")  # Ensure this file exists in your repo
    return df

df = load_data()

# Show dataset
st.subheader("📊 Dataset Preview")
st.write(df.head())

# Feature Selection
st.subheader("🔧 Feature Selection")
X = df.drop("Potability", axis=1)
y = df["Potability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Show Accuracy
st.subheader("📈 Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

st.subheader("💡 Try Custom Inputs for Prediction")
inputs = [st.number_input(f"{col}", value=0.0) for col in X.columns]

if st.button("Predict Potability"):
    prediction = model.predict([inputs])[0]
    result = "Safe to Drink 💧" if prediction == 1 else "Not Potable 🚱"
    st.subheader("🔍 Prediction Result")
    st.write(result)
