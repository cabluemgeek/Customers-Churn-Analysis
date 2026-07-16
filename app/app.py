import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="Bank Customer Churn Predictor", page_icon="📊", layout="centered")

st.title("📊 Bank Customer Churn Risk Predictor")
st.write(
    "Enter a customer's profile below to estimate their probability of churning "
    "(closing their account / leaving the bank), using an XGBoost model trained "
    "on ~10,000 historical customers."
)

# ------------------------------------------------------------
# Load the trained model + the exact feature schema it expects
# (both files must sit in the same folder as this app.py)
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "churn_model.pkl"))
    feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
    return model, feature_columns

model, feature_columns = load_model()

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
st.subheader("Customer profile")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 850, 650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 92, 40)
    tenure = st.slider("Tenure (years with the bank)", 0, 10, 5)

with col2:
    balance = st.number_input("Account Balance (€)", min_value=0.0, value=75000.0, step=1000.0)
    num_products = st.selectbox("Number of Products Held", [1, 2, 3, 4])
    has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary (€)", min_value=0.0, value=100000.0, step=1000.0)

predict_clicked = st.button("Predict Churn Risk", type="primary")

# ------------------------------------------------------------
# Build the input row EXACTLY the way the training data was built:
# same encodings, same engineered features, same column names.
# ------------------------------------------------------------
def build_input_row():
    has_card_flag = 1 if has_card == "Yes" else 0
    is_active_flag = 1 if is_active == "Yes" else 0
    gender_flag = 1 if gender == "Female" else 0   # matches the map used in training: Female -> 1, Male -> 0

    row = {
        "CreditScore": credit_score,
        "Gender": gender_flag,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_card_flag,
        "IsActiveMember": is_active_flag,
        "EstimatedSalary": salary,
        # one-hot geography, same as pd.get_dummies(..., drop_first=True) with France as baseline
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        # same two engineered features from Part 2 of the notebook
        "BalanceSalaryRatio": balance / (salary + 1),
        "EngagementScore": is_active_flag + has_card_flag,
    }

    input_df = pd.DataFrame([row])
    # reindex guarantees the columns are in the exact order the model was trained on;
    # fill_value=0 covers any dummy column not triggered by this particular input
    # (e.g. if geography = France, both Geography_Germany and Geography_Spain are already 0)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return input_df

# ------------------------------------------------------------
# Predict + display
# ------------------------------------------------------------
if predict_clicked:
    input_df = build_input_row()
    churn_probability = model.predict_proba(input_df)[0, 1]   # probability of class 1 (churn)

    if churn_probability < 0.30:
        risk_label, color = "Low Risk", "#1F8A87"
    elif churn_probability < 0.60:
        risk_label, color = "Medium Risk", "#E8664B"
    else:
        risk_label, color = "High Risk", "#B3261E"

    st.divider()
    st.subheader("Result")

    c1, c2 = st.columns(2)
    c1.metric("Churn Probability", f"{churn_probability:.1%}")
    c2.markdown(f"### <span style='color:{color}'>{risk_label}</span>", unsafe_allow_html=True)

    # simple gauge visual
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_probability * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#E8F5F4"},
                {"range": [30, 60], "color": "#FBE9E5"},
                {"range": [60, 100], "color": "#F8D7D5"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    revenue_at_risk = churn_probability * balance
    st.info(
        f"**Revenue at risk:** €{revenue_at_risk:,.0f} "
        f"(this customer's balance weighted by their churn probability)."
    )

    st.caption(
        "This tool estimates churn risk (likelihood of leaving the bank), not credit risk "
        "or creditworthiness -- those would come from a separately trained model."
    )
