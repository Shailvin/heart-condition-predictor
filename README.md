# heart-condition-predictor

A web-based application that predicts the likelihood of a heart condition based on user-provided medical data. This project uses Streamlit for the user interface and a Random Forest Classifier for machine learning predictions, enabling an interactive and intuitive experience for users.

# Project Overview
The Heart Condition Predictor is designed to:

Accept user inputs for key medical features (e.g., age, cholesterol levels, and blood pressure).
Use a trained machine learning model to classify whether the individual is at risk of a heart condition.
Provide a risk score along with the prediction for better insights.
Offer a clean and interactive web-based interface.
This project demonstrates practical implementation of data preprocessing, machine learning, and deployment using Streamlit Community Cloud.

# Features
Interactive Web Interface:
Built using Streamlit for a seamless user experience.
Sidebar inputs for medical data.
Real-time predictions with clear visualization of results.
Machine Learning:
Utilizes a Random Forest Classifier for classification.
Hyperparameter tuning performed using GridSearchCV for optimized performance.
Model Insights:
Displays model accuracy and AUC-ROC score to validate the predictions.
Customizable:
The app is flexible and can be extended with additional features or models.

Technologies Used
# Libraries
Python: The primary programming language.
Streamlit: For building the web app interface.
Scikit-learn: For machine learning and model evaluation.
Pandas and NumPy: For data manipulation and preprocessing.
Joblib: To save and load the trained machine learning model.
Tools
Streamlit Community Cloud: For deployment.
GitHub: For version control and code sharing

# Installation and Setup
Prerequisites
Python 3.8 or higher.
Internet connection.
Steps to Run the Project Locally
Clone the Repository:
git clone https://github.com/your-username/heart-condition-predictor.git
cd heart-condition-predictor

# Install Dependencies:

Ensure you have pip installed, then run: pip install -r requirements.txt

Run the Application:streamlit run heart_condition_app.py

Access the Application:

Open your browser and go to http://localhost:8501.

# Model Details
Algorithm: Random Forest Classifier.
Hyperparameter Tuning: GridSearchCV was used to optimize the following parameters:
Number of trees (n_estimators): 100, 200, 300.
Maximum tree depth (max_depth): None, 10, 20.
Minimum samples to split (min_samples_split): 2, 5, 10.
Scaling: Features were standardized using StandardScaler.

# Future Improvements
Add additional models (e.g., XGBoost, Logistic Regression) for comparison.
Implement user authentication for personalized predictions.
Include a "Save Results" feature to export predictions as a PDF or CSV.
Extend the app to support additional medical datasets for multi-disease prediction.

# Acknowledgments
The dataset used is from the Kaggle.
Thanks to the open-source Python and Streamlit communities for their invaluable tools and support.

# Contact
Feel free to reach out if you have any questions or suggestions:

Name: Shailvi

