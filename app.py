
# ─────────────────────────────────────────────────────────────────────────────
# Bank Customer Churn Predictor — Streamlit App
# Run: streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model    = joblib.load("best_churn_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    features = joblib.load("feature_names.pkl")
    return model, scaler, features

model, scaler, FEATURES = load_artefacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏦 Bank Customer Churn Risk Predictor")
st.markdown("Predict the probability that a customer will churn — and understand *why*.")
st.divider()

# ── Sidebar: Customer Input ───────────────────────────────────────────────────
st.sidebar.header("📋 Customer Profile")

credit_score    = st.sidebar.slider("Credit Score", 300, 850, 650)
age             = st.sidebar.slider("Age", 18, 95, 40)
tenure          = st.sidebar.slider("Tenure (years)", 0, 10, 3)
balance         = st.sidebar.number_input("Balance ($)", 0.0, 300000.0, 75000.0, step=1000.0)
num_products    = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=0)
has_cr_card     = st.sidebar.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
is_active       = st.sidebar.radio("Active Member?", ["Yes", "No"], horizontal=True)
salary          = st.sidebar.number_input("Estimated Salary ($)", 0.0, 300000.0, 100000.0, step=1000.0)
geography       = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender          = st.sidebar.selectbox("Gender", ["Male", "Female"])

# ── Build input dict ─────────────────────────────────────────────────────────
def build_features(credit_score, age, tenure, balance, num_products,
                   has_cr_card, is_active, salary, geography, gender):
    hcc = 1 if has_cr_card == "Yes" else 0
    iam = 1 if is_active   == "Yes" else 0

    row = {
        "CreditScore"          : credit_score,
        "Age"                  : age,
        "Tenure"               : tenure,
        "Balance"              : balance,
        "NumOfProducts"        : num_products,
        "HasCrCard"            : hcc,
        "IsActiveMember"       : iam,
        "EstimatedSalary"      : salary,
        # One-hot Geography
        "Geography_France"     : int(geography == "France"),
        "Geography_Germany"    : int(geography == "Germany"),
        "Geography_Spain"      : int(geography == "Spain"),
        # One-hot Gender
        "Gender_Female"        : int(gender == "Female"),
        "Gender_Male"          : int(gender == "Male"),
        # Engineered
        "BalanceToSalary"      : balance / (salary + 1),
        "ZeroBalance"          : int(balance == 0),
        "ProductsPerYear"      : num_products / (tenure + 1),
        "EngagementProducts"   : iam * num_products,
        "AgeTenure"            : age * tenure,
        "CreditScorePerAge"    : credit_score / age,
    }
    # Align to training feature order
    df = pd.DataFrame([row]).reindex(columns=FEATURES, fill_value=0)
    return df

input_df = build_features(credit_score, age, tenure, balance, num_products,
                          has_cr_card, is_active, salary, geography, gender)

# ── Predict ───────────────────────────────────────────────────────────────────
prob       = model.predict_proba(input_df)[0][1]
prediction = int(prob >= 0.5)

def risk_label(p):
    if p >= 0.75: return "🔴 HIGH RISK",   "error"
    if p >= 0.50: return "🟠 MEDIUM-HIGH", "warning"
    if p >= 0.30: return "🟡 MEDIUM RISK", "warning"
    return          "🟢 LOW RISK",    "success"

tier, msg_type = risk_label(prob)

# ── Output columns ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Probability", f"{prob*100:.1f}%")

with col2:
    st.metric("Prediction", "Will Churn" if prediction else "Will Stay")

with col3:
    st.metric("Risk Tier", tier)

if msg_type == "error":
    st.error(f"⚠️ This customer has a **high churn risk** ({prob*100:.1f}%). Immediate retention action recommended.")
elif msg_type == "warning":
    st.warning(f"This customer has a **moderate churn risk** ({prob*100:.1f}%). Consider a targeted offer.")
else:
    st.success(f"This customer is likely to stay. Churn probability: {prob*100:.1f}%.")

st.divider()

# ── Probability gauge ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 1.5))
ax.barh(["Churn Risk"], [prob],      color="#F44336" if prob>=0.5 else "#4CAF50", height=0.4)
ax.barh(["Churn Risk"], [1 - prob], left=[prob], color="#E0E0E0", height=0.4)
ax.set_xlim(0, 1)
ax.axvline(0.5, color="#555", linestyle="--", linewidth=1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_yticks([])
ax.set_title("Churn Probability Gauge", fontsize=11, fontweight="bold")
ax.text(prob, 0, f" {prob*100:.1f}%", va="center", fontweight="bold",
        color="white" if prob > 0.1 else "black")
fig.tight_layout()
st.pyplot(fig)

st.divider()
st.subheader("📋 Customer Feature Summary")
st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

st.divider()
st.markdown("""
### 📌 Retention Recommendations
| Risk Tier | Suggested Action |
|-----------|------------------|
| 🔴 High   | Urgent personal outreach, exclusive loyalty reward, dedicated relationship manager |
| 🟠 Med-High | Personalised product offer, fee waiver, upgrade invitation |
| 🟡 Medium | Re-engagement email, satisfaction survey, product cross-sell |
| 🟢 Low    | Standard newsletter, routine check-in |
""")
