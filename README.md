# Real AI Application — Credit Card Default Prediction

## Weekly Task
This project applies the full ML workflow to a real credit-risk classification problem:
data cleaning → analysis → feature engineering → model training → prediction →
result interpretation → stakeholder presentation.

## Dataset
**Default of Credit Card Clients** from the UCI Machine Learning Repository.
The dataset contains 30,000 instances and 23 explanatory features, with a binary target
for default payment next month.

Official source: https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients

## Folder structure

```text
real_ai_application/
├── app.py
├── requirements.txt
├── README.md
├── task_week9_credit_default.ipynb
├── data/
│   └── default_of_credit_card_clients.csv

```

## 1. Put the data in the project
Copy your CSV from VS Code into:

`data/default_of_credit_card_clients.csv`

The Streamlit app also supports the original Excel filename:
`data/default of credit card clients.xls`

## 2. Install packages

```bash
python -m pip install -r requirements.txt
```

## 3. Run the notebook
Open `task_week9_credit_default.ipynb` and run all cells.
The notebook performs EDA, feature engineering, trains Random Forest,
evaluates it, and saves the model pipeline.

## 4. Run the Streamlit application

```bash
streamlit run app.py
```

## Stakeholder message
The application is intended as a **decision-support tool**. It estimates the
probability that a client will default next month. A high-risk result should
trigger additional review rather than an automatic rejection.


