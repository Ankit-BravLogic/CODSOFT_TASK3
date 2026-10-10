# Customer Churn Prediction | CodSoft Machine Learning Internship

An end-to-end Machine Learning project that predicts whether a bank customer is likely to leave the bank. The application uses a trained Gradient Boosting classifier and an interactive Streamlit interface to generate predictions from customer details.

## 🚀 Live Demo

**Try the deployed application:**  
https://customer-churn-prediction-9.streamlit.app/

## 📂 GitHub Repository

**Source code:**  
https://github.com/Ankit-BravLogic/CODSOFT_TASK3

## 📌 Project Overview

Customer churn occurs when customers stop using a company's services. Predicting churn helps businesses identify customers who may leave and develop better customer-retention strategies.

This project uses historical bank customer data to predict the likelihood of churn.

## 🎯 Objectives

- Analyze customer data and identify churn patterns.
- Perform exploratory data analysis (EDA).
- Preprocess data for machine learning.
- Train and compare classification models.
- Evaluate model performance using appropriate metrics.
- Build an interactive web application using Streamlit.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Git and GitHub

## 📊 Dataset

The project uses `Churn_Modelling.csv`, containing 10,000 customer records and 14 columns.

**Target variable:** `Exited`

- `0` — Customer stayed.
- `1` — Customer left.

## 🤖 Machine Learning Model

The project compares classification models and uses **Gradient Boosting** as the final model.

### Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 86.90% |
| Precision | 78.21% |
| Recall | 49.39% |
| F1-score | 60.54% |
| ROC-AUC | 87.02% |

These scores are based on the project's held-out test dataset.

## ✨ Application Features

- Interactive customer input form.
- Churn prediction for individual customers.
- Simple web interface built with Streamlit.
- Predictions using the trained machine learning model.

## 📁 Project Structure

```text
CODSOFT_TASK3/
├── app.py
├── check_data.py
├── eda.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── Churn_Modelling.csv
├── models/
│   ├── churn_model.pkl
│   └── model_comparison.csv
└── reports/
    └── figures/
        ├── churn_by_activity.png
        ├── churn_by_age.png
        ├── churn_by_geography.png
        └── churn_distribution.png
```

## ⚙️ Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Ankit-BravLogic/CODSOFT_TASK3.git
cd CODSOFT_TASK3
```

### 2. Create and activate a virtual environment

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

**Note:** Use a Python version compatible with the pinned scikit-learn version and the saved model.

## 📈 Exploratory Data Analysis

The project includes visualizations of:

- Customer churn distribution.
- Churn by age.
- Churn by geography.
- Churn by account activity.

The analysis helps identify patterns associated with customer churn.

## 🔮 Future Improvements

- Improve recall for customers who are likely to churn.
- Add model explainability using feature importance or SHAP.
- Provide personalized customer-retention recommendations.
- Improve the user interface and add further evaluation insights.

## 👨‍💻 Author

**Ankit Kumar Verma**

B.Tech Student | Machine Learning and AI Enthusiast

## 🙌 Acknowledgement

This project was developed as part of the **CodSoft Machine Learning Internship — Task 3: Customer Churn Prediction**.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
