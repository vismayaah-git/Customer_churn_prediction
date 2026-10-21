# Customer Churn Prediction

## Project Overview

This project uses machine learning to predict whether a telecommunications customer is likely to leave the service (churn). It includes a Jupyter notebook for data analysis, model training, and evaluation, along with a Streamlit app for making predictions from customer details.

## Dataset

The project uses the Telco Customer Churn dataset in `data/Telco-Customer-Churn.csv`. It contains customer demographics, subscribed phone and internet services, contract and payment information, monthly charges, total charges, and the `Churn` target.

## Machine Learning

The notebook compares Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and Gradient Boosting classifiers. Models are evaluated on a held-out test set using accuracy, precision, recall, F1 score, confusion matrices, and classification reports. The notebook also demonstrates five-fold cross-validation and hyperparameter tuning with `GridSearchCV`.

The Streamlit app uses the saved Random Forest model in `churn_model.pkl` and the fitted preprocessing pipeline in `preprocessor.pkl`. The preprocessor scales numerical features and one-hot encodes categorical features.

## Streamlit App

The app collects customer and service details, applies the saved preprocessing pipeline, and displays the model's churn prediction. A prediction is an estimate, not a guarantee of future customer behavior.

### Run locally

1. Clone the repository and open a terminal in the project folder.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run streamlit_app.py
   ```

4. Open the local URL printed in the terminal.

## Technologies

Python, pandas, NumPy, scikit-learn, joblib, Streamlit, and JupyterLab.

## Project Structure

```text
Customer_churn_prediction/
├── streamlit_app.py
├── customer_churn.ipynb
├── churn_model.pkl
├── preprocessor.pkl
├── requirements.txt
├── README.md
└── data/
    └── Telco-Customer-Churn.csv
```
