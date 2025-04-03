import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# App Title & Description
st.title("💧 Water Potability Prediction App")
st.markdown("Predict whether water is **potable (safe to drink)** based on chemical properties!")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")  # Ensure this file is in your repo
    return df

df = load_data()

# Data Cleaning - Handling Missing Values
df.fillna(df.median(), inplace=True)  # Fill missing values with median

# Show Dataset
st.subheader("📊 Dataset Preview")
st.write(df.head())

# Feature Selection
X = df.drop("Potability", axis=1)
y = df["Potability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
st.subheader("⚙️ Choose a Model for Prediction")
model_option = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "KNN"])

if model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif model_option == "Logistic Regression":
    model = LogisticRegression()
elif model_option == "KNN":
    model = KNeighborsClassifier(n_neighbors=5)

# Train Model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Show Accuracy
st.subheader("📈 Model Performance")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Feature Importance for Random Forest
if model_option == "Random Forest":
    st.subheader("🔥 Feature Importance")
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=feature_importance[sorted_idx], y=X.columns[sorted_idx], palette="Blues_r")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance in Random Forest")
    st.pyplot(plt)

# User Input for Prediction
st.subheader("💡 Try Custom Inputs for Prediction")
inputs = [st.number_input(f"{col}", value=0.0) for col in X.columns]

if st.button("Predict Potability"):
    prediction = model.predict([inputs])[0]
    result = "✅ Safe to Drink 💧" if prediction == 1 else "❌ Not Potable 🚱"
    st.subheader("🔍 Prediction Result")
    st.write(result)
