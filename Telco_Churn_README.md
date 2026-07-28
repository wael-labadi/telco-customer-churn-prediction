# Telco Customer Churn — Data Cleaning & Exploratory Data Analysis

A data science project analyzing customer churn for a telecom company: understanding the data, cleaning it, exploring it visually, and preparing it for machine learning. This is the first stage of an ongoing project — model training is the next phase.

## Project Description

Customer churn (customers leaving a service) is one of the most common and costly problems telecom companies face. This project explores a real telecom customer dataset to understand *why* customers leave, clean the data into a usable state, and surface early patterns before building a predictive model in the next phase.

**Question the project aims to answer:** Which customer characteristics and account details are associated with churn, and what does the data look like once it's properly cleaned and prepared for modeling?

**Project type:** Classification (target: `Churn` — Yes/No)

**Target variable:** `Churn`

**Features:** Customer demographics (gender, senior citizen status, partner/dependents), account information (tenure, contract type, payment method, billing), and subscribed services (phone, internet, streaming, tech support, etc.)

## Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 original columns
- Raw file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Cleaned/encoded file: `Telco_Customer_Churn_Cleaned.csv`

## Data Preprocessing

The full cleaning pipeline, in order:

1. **Explored the raw data** with `.head()`, `.info()`, and `.isnull().sum()` to understand structure and spot issues
2. **Fixed `TotalCharges`** — it was stored as text (`object`) instead of a number. Found 11 rows with blank/whitespace values (all new customers with `tenure = 0`, meaning they hadn't been billed yet), converted the column to numeric with `pd.to_numeric(errors='coerce')`, then filled the resulting NaNs with `0` since those customers genuinely have $0 total charges so far
3. **Checked for duplicate rows** — confirmed none remained after cleaning
4. **Dropped `customerID`** — a unique identifier with no predictive value for modeling
5. **Reviewed all text/categorical columns** (`.unique()` on each) to confirm consistent labeling with no typos or stray values
6. **Checked numeric columns** with `.describe()` for unrealistic values or outliers — none found
7. **Encoded the target:** mapped `Churn` from Yes/No to 1/0
8. **One-hot encoded** all remaining categorical columns (`pd.get_dummies(drop_first=True)`) to prepare the dataset for model training

## Exploratory Data Analysis — Key Insights

**1. The dataset is imbalanced.**
About 73% of customers did not churn vs. 27% who did. This matters directly for the next phase: a model could predict "No" every time and still look highly accurate, so evaluation will need Precision, Recall, and F1-score rather than accuracy alone.

**2. Tenure and TotalCharges are strongly correlated (0.83), with a moderate link between MonthlyCharges and TotalCharges (0.65).**
Makes intuitive sense — the longer a customer stays, the more they've paid in total. This is a multicollinearity warning sign; one of these features may need to be dropped or combined before modeling.

**3. Customer tenure is bimodal, not normally distributed.**
There are two large groups: brand-new customers (0–5 months) and long-term loyal customers (65–72 months), with fewer customers in between. This suggests two distinct customer segments — newer customers who may be more likely to churn, and long-term customers who are more likely to stay.

## Visualizations

- **Churn count plot** — shows the class imbalance described above
- **Correlation heatmap** (numeric features) — shows the tenure/TotalCharges/MonthlyCharges relationship
- **Tenure histogram** — shows the bimodal distribution of customer tenure

## Tech Stack

- Python 3, Google Colab
- **Pandas / NumPy** — data loading and cleaning
- **Seaborn / Matplotlib** — visualization

## Project Status

- [x] Data understanding
- [x] Data cleaning (missing values, duplicates, data types, outlier check)
- [x] Exploratory visualizations with interpretation
- [x] Dataset encoded and prepared for model training
- [ ] Model training (Part 2 — in progress)
- [ ] Model evaluation

## Files

| File | Description |
|---|---|
| `telco_customer_churn.py` / `.ipynb` | Full cleaning + EDA notebook |
| `Telco_Customer_Churn_Cleaned.csv` | Cleaned, encoded dataset ready for modeling |

## Author

Part of an AI & Machine Learning course — Data Science module, weekly project assignment.
