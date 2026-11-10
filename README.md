# Customer Churn Prediction

Predicts whether a telecommunications customer may churn using the Telco Customer Churn dataset.

## Models

The notebook compares Logistic Regression, Decision Tree, Random Forest, KNN, and Gradient Boosting and Model Evaluvations. Tuned Random Forest hyperparameters with GridSearchCV and 5-fold cross-validation.


- Best accuracy search: **80% accuracy**; **49% churn recall**
- Best churn F1 search: **74% accuracy**; **73% churn recall**

The app uses the saved model and preprocessor. Higher churn recall means catching more churners, but may also produce more false alerts.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files

- `customer_churn.ipynb` — analysis, training, and evaluation
- `streamlit_app.py` — prediction app
- `data/Telco-Customer-Churn.csv` — dataset
- `churn_model.pkl`, `preprocessor.pkl` — saved model artifacts
