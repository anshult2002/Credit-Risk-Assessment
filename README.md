# Credit Risk Assessment

A machine learning project for predicting **loan default risk** using borrower and loan-related information. The project compares multiple classification models, handles class imbalance, performs hyperparameter tuning, and uses SHAP for model interpretability.

## Overview

The goal is to predict whether a borrower is likely to **default on a loan** based on financial and personal attributes.

The project includes:

* Data preprocessing for numerical and categorical features
* Missing-value handling
* Feature scaling and one-hot encoding
* Class imbalance handling
* Logistic Regression, Random Forest, and XGBoost
* Hyperparameter tuning with RandomizedSearchCV
* Model evaluation using classification metrics
* SHAP-based model interpretation

## Dataset

The dataset contains **19K+ loan applications** with borrower and loan characteristics.

### Features

Some of the important features include:

* `person_age` — Borrower's age
* `person_income` — Annual income
* `person_home_ownership` — Home ownership status
* `person_emp_length` — Employment length
* `loan_intent` — Purpose of the loan
* `loan_grade` — Loan grade
* `loan_amnt` — Loan amount
* `loan_int_rate` — Interest rate
* `loan_percent_income` — Loan amount as a percentage of income
* `cb_person_default_on_file` — Previous credit default indicator
* `cb_person_cred_hist_length` — Length of credit history

### Target

```text
loan_status
```

where:

```text
0 → No default
1 → Default
```

## Data Preprocessing

The preprocessing pipeline handles numerical and categorical features separately.

### Numerical Features

* Missing-value imputation using the median
* Feature scaling using StandardScaler

### Categorical Features

* Missing-value handling
* One-hot encoding
* Unknown categories handled during inference

The preprocessing is implemented using Scikit-learn pipelines and `ColumnTransformer`.

## Handling Class Imbalance

Loan default is an imbalanced classification problem.

Two approaches were used:

* **Class weighting**
* **SMOTENC** for generating synthetic samples while accounting for categorical features

Resampling is applied only to the training data, while the test set remains unchanged for unbiased evaluation.

## Models

The following classification algorithms were evaluated:

### Logistic Regression

Used as the baseline linear classification model and later tuned using RandomizedSearchCV.

### Random Forest

An ensemble of decision trees used to capture nonlinear relationships between borrower and loan features.

### XGBoost

A gradient-boosted tree model used to model more complex relationships in the data.

## Hyperparameter Tuning

`RandomizedSearchCV` with **5-fold cross-validation** was used to search over **150 hyperparameter configurations**.

The primary optimization metric was:

```text
Recall
```

Recall was emphasized because missing a borrower who is actually likely to default represents a false negative.

## Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The final evaluation is performed on a **held-out test set that is not resampled**.

## Results

The tuned **class-weighted Logistic Regression** achieved:

| Metric   |    Score |
| -------- | -------: |
| Recall   | **0.80** |
| Accuracy | **0.80** |

The model comparison also included Random Forest and XGBoost to evaluate whether nonlinear models provided improvements over the linear baseline.

## Model Interpretability

SHAP was used to understand which features contributed most to the model's predictions.

The analysis helps answer questions such as:

* Which borrower characteristics are associated with higher default risk?
* Which features contribute most to individual predictions?
* In which direction does a feature influence the prediction?

For the Logistic Regression model, SHAP's linear explainer is used to calculate feature contributions.

## SHAP Analysis

A SHAP summary plot is used to visualize global feature importance.

Mean absolute SHAP values are used to rank features according to their overall contribution to model predictions.

```text
Feature
   │
   ├── Feature A  ███████████
   ├── Feature B  █████████
   ├── Feature C  ███████
   └── Feature D  █████
```

## Project Workflow

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Train / Test Split
     │
     ├──────────────► Test Set
     │                  │
     ▼                  │
Preprocessing            │
     │                  │
     ▼                  │
Class Imbalance          │
Handling                 │
     │                  │
     ▼                  │
Baseline Models          │
     │                  │
     ▼                  │
RandomizedSearchCV       │
     │                  │
     ▼                  │
Best Model ──────────────┘
     │
     ▼
Evaluation
     │
     ▼
SHAP Interpretation


## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* imbalanced-learn
* SHAP
* Matplotlib
* Seaborn

## Key Takeaways

* Built a complete credit-risk classification pipeline for **19K+ loan applications**.
* Compared linear and tree-based classification models.
* Addressed class imbalance using **class weighting and SMOTENC**.
* Used **5-fold RandomizedSearchCV with 150 configurations** for hyperparameter tuning.
* Achieved **0.80 recall and 0.80 accuracy** with the class-weighted Logistic Regression model.
* Used **SHAP** to interpret model predictions and identify influential credit-risk factors.

## Author

**Anshul Thakur**

If you found this project useful, consider giving the repository a ⭐.
