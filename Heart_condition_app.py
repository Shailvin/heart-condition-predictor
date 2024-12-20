import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import joblib

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("heart.csv")
    return data

# Preprocess the data
def preprocess_data(data):
    X = data.drop(['target'], axis=1)
    y = data['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# Train the model
@st.cache_resource
def train_model(X, y):
    # Random Forest with GridSearch for tuning
    param_grid = {'n_estimators': [100, 200, 300],
                  'max_depth': [None, 10, 20],
                  'min_samples_split': [2, 5, 10]}
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, "Heart_Predict_RF.joblib")
    return best_model

# Predict function
def predict(model, scaler, user_input):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[:, 1]
    return prediction, prob

# Main Streamlit App
def main():
    st.title("Heart Condition Predictor 💖")
    st.write("This app predicts the likelihood of having a heart condition based on input features.")

    # Load data and preprocess
    data = load_data()
    X, y, scaler = preprocess_data(data)

    # Train the model
    model = train_model(X, y)

    # Input fields for user
    st.sidebar.header("Input Features")
    age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.sidebar.selectbox("Sex", [1, 0])
    cp = st.sidebar.slider("Chest Pain Type", 0, 3, 1)
    trestbps = st.sidebar.number_input("Resting Blood Pressure", value=120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0])
    restecg = st.sidebar.slider("Resting ECG Results", 0, 2, 1)
    thalach = st.sidebar.number_input("Max Heart Rate Achieved", value=150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [1, 0])
    oldpeak = st.sidebar.number_input("ST Depression", value=1.0)
    slope = st.sidebar.slider("Slope of Peak Exercise ST", 0, 2, 1)
    ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.sidebar.slider("Thalassemia (0-2)", 0, 2, 1)

    # Predict heart condition
    user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    if st.button("Predict"):
        prediction, probability = predict(model, scaler, user_input)
        if prediction[0] == 1:
            st.error(f"🚨 The person is at risk of a heart condition. Risk Score: {probability[0]:.2f}")
        else:
            st.success(f"✅ The person is unlikely to have a heart condition. Risk Score: {probability[0]:.2f}")
        
    # Display model performance
    st.subheader("Model Performance")
    st.text("Accuracy on Training Data: {:.2f}".format(accuracy_score(y, model.predict(X))))
    st.text("ROC-AUC Score: {:.2f}".format(roc_auc_score(y, model.predict_proba(X)[:, 1])))

    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(data.head())

if __name__ == "__main__":
    main()
