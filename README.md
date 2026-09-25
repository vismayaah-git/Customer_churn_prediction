# Customer Churn Prediction

## Project Overview

This project predicts whether a telecom customer is likely to churn using Machine Learning.

The project includes data cleaning, exploratory data analysis, preprocessing, multiple classification algorithms, model evaluation, cross-validation, hyperparameter tuning, feature importance analysis, and Flask deployment.

## Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information such as:

- Customer demographics
- Tenure
- Phone and internet services
- Contract type
- Payment method
- Monthly charges
- Total charges

The target variable is `Churn`.

## Machine Learning Models

The following classification algorithms were tested:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Gradient Boosting

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- 5-Fold Cross-Validation

Hyperparameter tuning was performed using `GridSearchCV`.

## Data Preprocessing

The project includes:

- Missing-value handling
- Numerical feature scaling
- Categorical feature encoding using One-Hot Encoding
- Train-test splitting

## Deployment

The trained model was saved using `joblib` and deployed using Flask.

The web application allows users to enter customer information and receive a churn prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- Joblib
- JupyterLab
- HTML

## Project Structure

```text
Customer_churn_prediction/
│
├── app.py
├── customer_churn.ipynb
├── churn_model.pkl
├── preprocessor.pkl
├── requirements.txt
├── README.md
│
├── data/
│   └── Telco-Customer-Churn.csv
│
└── templates/
    └── index.html

## How to Run

1. Install the required libraries
   pip install -r requirements.txt    

2. Run the Flask application
   python app.py

3. Open the application
   Open this address in your browser: http://127.0.0.1:5000

##Project Outcome

The final application provides a web interface where customer details can be entered and the trained Machine Learning model predicts whether the customer is likely to churn.   