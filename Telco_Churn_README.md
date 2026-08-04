# Telco Customer Churn — Prediction Project (Data Cleaning, EDA & Machine Learning)

An end-to-end data science project analyzing customer churn for a telecom company: understanding the data, cleaning it, exploring it visually, and training a machine learning model to predict which customers are likely to leave.

## Project Description

Customer churn (customers leaving a service) is one of the most common and costly problems telecom companies face. This project explores a real telecom customer dataset to understand *why* customers leave, cleans the data into a usable state, surfaces early patterns through exploratory analysis, and then trains a classification model to predict churn.

**Question the project aims to answer:** Which customer characteristics and account details are associated with churn, and can we accurately predict which customers are at risk of leaving?

**Project type:** Classification (target: `Churn` — Yes/No)

**Target variable:** `Churn`

**Features:** Customer demographics (gender, senior citizen status, partner/dependents), account information (tenure, contract type, payment method, billing), and subscribed services (phone, internet, streaming, tech support, etc.)

## Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 original columns
- Raw file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Cleaned/encoded file: `Telco_Customer_Churn_Cleaned.csv`

## Part 1 — Data Preprocessing

The full cleaning pipeline, in order:

1. **Explored the raw data** with `.head()`, `.info()`, and `.isnull().sum()` to understand structure and spot issues
2. **Fixed `TotalCharges`** — it was stored as text (`object`) instead of a number. Found 11 rows with blank/whitespace values (all new customers with `tenure = 0`, meaning they hadn't been billed yet), converted the column to numeric with `pd.to_numeric(errors='coerce')`, then filled the resulting NaNs with `0` since those customers genuinely have $0 total charges so far
3. **Checked for duplicate rows** — confirmed none remained after cleaning
4. **Dropped `customerID`** — a unique identifier with no predictive value for modeling
5. **Reviewed all text/categorical columns** (`.unique()` on each) to confirm consistent labeling with no typos or stray values
6. **Checked numeric columns** with `.describe()` for unrealistic values or outliers — none found
7. **Encoded the target:** mapped `Churn` from Yes/No to 1/0
8. **One-hot encoded** all remaining categorical columns (`pd.get_dummies(drop_first=True)`) to prepare the dataset for model training

## Part 1 — Exploratory Data Analysis: Key Insights

**1. The dataset is imbalanced.**
About 73% of customers did not churn vs. 27% who did. This directly shaped how the model was evaluated in Part 2 — accuracy alone would be misleading, so Precision, Recall, F1-score, and AUC were used instead.

**2. Tenure and TotalCharges are strongly correlated (0.83), with a moderate link between MonthlyCharges and TotalCharges (0.65).**
Makes intuitive sense — the longer a customer stays, the more they've paid in total. This was a multicollinearity warning sign going into modeling.

**3. Customer tenure is bimodal, not normally distributed.**
There are two large groups: brand-new customers (0–5 months) and long-term loyal customers (65–72 months), with fewer customers in between. This suggests two distinct customer segments — newer customers who are more likely to churn, and long-term customers who are more likely to stay. This pattern shows up again in Part 2's feature importance results.

## Part 2 — Model Training

**Algorithm used:** Logistic Regression

Chosen as a strong, interpretable baseline for binary classification — it's a standard starting point for churn prediction, and its coefficients directly show which features increase or decrease churn risk (see Feature Importance below).

**Pipeline:**
1. Split features (`X`) from the target (`y = Churn`)
2. Train/test split (80/20), using `stratify=y` to preserve the 73/27 class balance in both sets
3. Scaled features with `StandardScaler` (fit on training data only, to avoid data leakage)
4. Trained a `LogisticRegression` model on the scaled training data
5. Generated both class predictions (0/1) and churn probabilities on the test set

### Model Evaluation

| Metric | Score |
|---|---|
| Accuracy | *[fill in]* |
| Precision | *[fill in]* |
| Recall | *[fill in]* |
| F1 Score | *[fill in]* |
| AUC | *[fill in]* |

**Why Recall matters most here:** with an imbalanced dataset, a model that just predicts "No Churn" every time would still score ~73% accuracy without learning anything. Recall — how many actual churners the model successfully catches — is the more meaningful metric for this business problem, since missing an at-risk customer (a false negative) is costlier than a false alarm.

**Confusion Matrix and ROC Curve** were used to visualize model performance beyond a single accuracy number (see notebook for plots).

### Feature Importance

The model's coefficients were extracted to identify which features most strongly influence churn risk (e.g., contract type, tenure, payment method) — providing actionable business insight, not just a prediction.

### Example Prediction

Tested the trained model on a new, hypothetical high-risk customer profile (short tenure, month-to-month contract, no add-on services, electronic check payment) — the model predicted **Churn = Yes**, with an **82.44% probability**, consistent with the patterns found during EDA.

## Visualizations

- Churn count plot (class imbalance)
- Correlation heatmap (numeric features)
- Tenure histogram (bimodal distribution)
- Confusion matrix (model performance breakdown)
- ROC curve (model discrimination ability)

## Tech Stack

- Python 3, Google Colab
- **Pandas / NumPy** — data loading and cleaning
- **Seaborn / Matplotlib** — visualization
- **Scikit-learn** — model training and evaluation (`LogisticRegression`, `StandardScaler`, `train_test_split`, classification metrics)

## Project Status

- [x] Data understanding
- [x] Data cleaning (missing values, duplicates, data types, outlier check)
- [x] Exploratory visualizations with interpretation
- [x] Dataset encoded and prepared for model training
- [x] Model training (Logistic Regression)
- [x] Model evaluation (Accuracy, Precision, Recall, F1, AUC, Confusion Matrix, ROC Curve)
- [ ] *(optional next step)* Compare against other models (e.g., Random Forest, XGBoost)

## Files

| File | Description |
|---|---|
| `telco_customer_churn.py` / `.ipynb` | Data cleaning + EDA notebook (Part 1) |
| `telco_churn_model.py` / `.ipynb` | Model training + evaluation notebook (Part 2) |
| `Telco_Customer_Churn_Cleaned.csv` | Cleaned, encoded dataset used for modeling |

## Author

Part of an AI & Machine Learning course — Data Science module, weekly project assignment.
