# Customer Churn Prediction — CodSoft Task 3

## Project Overview

This project predicts whether a bank customer is likely to leave the bank using machine learning. It was developed as part of the **CodSoft Machine Learning Internship — Task 3**.

## Objectives

- Analyze historical customer data.
- Explore customer churn patterns.
- Train and compare machine learning classification models.
- Provide an interactive interface for customer churn prediction.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset

The project uses `Churn_Modelling.csv`, containing 10,000 customer records and 14 columns. The target column is `Exited`:

- `0` — Customer stayed.
- `1` — Customer left.

Identifier columns such as `RowNumber`, `CustomerId`, and `Surname` are excluded from model training.

## Machine Learning Models

The following models were evaluated:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Gradient Boosting achieved the highest F1 score among the evaluated models.

## Evaluation Results

| Metric | Gradient Boosting |
|---|---:|
| Accuracy | 86.90% |
| Precision | 78.21% |
| Recall | 49.39% |
| F1 Score | 0.6054 |
| ROC-AUC | 0.8702 |

The model's recall indicates that further improvements may be needed to identify more customers who churn.

## Run Locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Run the application:

   `streamlit run app.py`

## Project Structure

- `app.py` — Streamlit interface
- `train_model.py` — Model training and evaluation
- `eda.py` — Exploratory data analysis
- `data/` — Dataset
- `models/` — Saved model and model comparison results
- `reports/figures/` — EDA charts
- `requirements.txt` — Python dependencies

## Disclaimer

This model provides estimates based on historical data and should not be treated as a guarantee of future customer behavior.

## Internship

Developed for CodSoft Machine Learning Internship — Task 3.