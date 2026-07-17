# Bank Customer Churn Analysis

Predicting **who** will churn, explaining **why**, and modeling **when** — combining a classification model, SHAP explainability, and survival analysis to turn churn prediction into an actionable retention tool for a retail bank.

**[Live churn risk predictor](https://customers-churn-analysis-5ffpxtupaphbftbmmpnu2y.streamlit.app/)** · **[Interactive dashboard](https://cabluemgeek.github.io/Customers-Churn-Analysis/interactive-dash.html)** · **[Full notebook](churn-analysis/Churn_Analysis.html)**

---

## Business problem

Customer acquisition costs far more than retention, but most churn models stop at "will this customer leave?" — which tells a bank *who* to worry about but not *when* to act or *how much* is actually at stake. This project answers all three:

- **Who** is likely to churn (classification)
- **Why** — which factors actually drive the decision (SHAP + Cox hazard ratios)
- **When** in the customer relationship risk peaks (survival analysis)
- **How much** revenue is at risk, ranked by customer (business layer)

## Dataset

- **Source:** Kaggle "Bank Customer Churn" dataset
- **Size:** 10,000 customers, 2,037 observed churn events (~20% churn rate)
- **Features:** CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- **Target:** `Exited` (1 = churned, 0 = retained)

## Repository structure

```
Customers-Churn-Analysis/
├── Churn_Modelling.csv              # raw dataset          
├── Dashboard_items/                  # individual Plotly chart exports
│   ├── confusion_matrix.html
│   ├── roc_curve.html
│   ├── correlation_heatmap.html
│   ├── age_distribution.html
│   ├── churn_by_category.html
│   └── churn_distribution.html
between the 6 charts above
├── app/                               # Streamlit churn predictor
│   ├── app.py
│   ├── requirements.txt
│   ├── churn_model.pkl
│   └── feature_columns.pkl
├── Customers_Churn_Analysis.ipynb             # full exported notebook (code + commentary) 
├── interactive-dash.html             # dashboard shell that toggles 
├── priority_retention_list.csv       # top 10% highest-risk customers, ranked by revenue at risk
├── requirements.txt             
└── README.md
```

## Methodology

**1. Exploratory Data Analysis**
Churn rate broken down by geography, activity status, product count, age, and correlation between numeric features. Germany shows a notably higher churn rate than France/Spain; inactive members and customers holding 3-4 products churn far more than average.

**2. Feature engineering**
- One-hot encoding for Geography, binary encoding for Gender
- `BalanceSalaryRatio` — balance relative to income, a disengagement signal
- `EngagementScore` — combined activity + card ownership signal

**3. Classification**
Three models compared: Logistic Regression (interpretable baseline), Random Forest, and XGBoost — evaluated with AUC, precision/recall, and a confusion matrix rather than plain accuracy, since the target is imbalanced.

**4. Explainability**
SHAP values on the XGBoost model to identify and rank the true drivers of individual predictions, not just global feature importance.

**5. Survival analysis**
Kaplan-Meier curves (overall and stratified by product count) plus a Cox Proportional Hazards model, using `Tenure` as duration and `Exited` as the event flag — modeling *when* risk peaks, not just *whether* it occurs.

**6. Business layer**
Churn probability weighted by each customer's balance to estimate revenue at risk, ranked into a prioritized retention list for the top 10% highest-risk customers.

## Key results

| Model | AUC | Recall (churn class) |
|---|---|---|
| Logistic Regression | — | — |
| Random Forest | — | — |
| **XGBoost** | **0.868** | **0.744** |

**Cox Proportional Hazards** (n=10,000, 2,037 events, concordance = 0.70):

| Feature | Hazard ratio (exp(coef)) | Interpretation |
|---|---|---|
| IsActiveMember | 0.46 | Active members face 54% lower churn hazard |
| Age | 1.05 | Each additional year increases hazard by 5% |
| Balance | 1.00 | Statistically significant but negligible per-unit effect |
| CreditScore | 1.00 | Statistically significant but negligible per-unit effect |
| NumOfProducts (linear) | 0.98 | Not significant — effect is non-linear (see notebook) |

The linear NumOfProducts term understates its real impact: Kaplan-Meier stratification and SHAP both show a sharp non-linear risk spike at 3-4 products that a single linear coefficient can't capture. Binning it into a categorical flag (1-2 vs 3-4 products) recovers a significant effect — documented in the notebook as a methodological finding in its own right.

## Live tools

- **[Churn risk predictor](https://customers-churn-analysis-5ffpxtupaphbftbmmpnu2y.streamlit.app/)** — enter a customer's profile and get a real-time churn probability, risk label, and revenue-at-risk estimate from the trained XGBoost model.
- **[Interactive dashboard](https://cabluemgeek.github.io/Customers-Churn-Analysis/interactive-dash.html)** — toggle between all 6 analysis charts (confusion matrix, ROC curve, correlation heatmap, age distribution, churn by category, churn distribution).

## Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · lifelines (survival analysis) · Plotly · Streamlit

## Running locally

```bash
# clone the repo
git clone https://github.com/cabluemgeek/Customers-Churn-Analysis.git
cd Customers-Churn-Analysis

# run the notebook
jupyter notebook churn-analysis/  # or open Churn_Analysis.html directly, no execution needed

# run the predictor app locally
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Author

**Kamélia Zenati**
[Portfolio](https://cabluemgeek.github.io) · [GitHub](https://github.com/cabluemgeek)