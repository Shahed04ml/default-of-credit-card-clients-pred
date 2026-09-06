# Short Presentation — Credit Card Default Prediction

## Slide 1 — Problem
**Goal:** predict whether a credit-card client will default on the next payment.

Business value:
- identify higher-risk clients earlier;
- support credit-risk review;
- turn historical customer data into a practical prediction tool.

## Slide 2 — Data
Dataset: **Default of Credit Card Clients**.

- 30,000 client records
- 23 explanatory features
- demographic information
- credit limit
- repayment history
- bill amounts
- previous payment amounts
- binary target: default payment next month (0/1)

## Slide 3 — Data Analysis
Checks performed:
- dataset shape and data types
- descriptive statistics
- missing values
- duplicate rows
- target-class balance

Visual analysis:
- default vs no-default class balance
- credit-limit distribution by default status
- default rate by age group
- correlation of repayment history with default

## Slide 4 — Feature Engineering
New features:
1. `UTILIZATION_1` — recent bill / credit limit
2. `UTILIZATION_2` — previous bill / credit limit
3. `PAY_RATIO_1` — payment / previous bill
4. `TOTAL_DELAY_MONTHS` — count of delayed-payment months
5. `BILL_DIFF` — change between two recent bills

These features make the raw financial behavior easier for the model to use.

## Slide 5 — Machine Learning
Model: **Random Forest Classifier**

Main settings:
- 150 trees
- max depth = 10
- balanced class weights
- stratified 80/20 train-test split
- StandardScaler for numeric variables
- OneHotEncoder for categorical variables

Evaluation:
- accuracy
- classification report
- ROC-AUC
- confusion matrix

## Slide 6 — Application
The Streamlit interface lets a user enter:
- credit limit
- age
- demographic categories
- repayment history
- recent bills
- previous payments

The app returns:
- predicted class
- probability of default
- a simple business interpretation

## Slide 7 — Conclusion
**What we built:** a complete ML workflow from raw data to an interactive application.

**Key message:** machine learning can help prioritize credit-risk review, but
the prediction should be treated as decision support rather than an automatic
financial decision.


