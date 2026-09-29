"""
Loan Default Risk Prediction — Demo Interface
ALU Capstone — Initial Software Product Demonstration (ML Track)
Author: Henriette Utatsineza 

Run locally with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Default Risk Predictor", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (exported from the Colab notebook as xgboost_model_A.json)
# Place that file in the same folder as this script before running.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("xgboost_model_A.json")
    return model

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

FEATURE_NAMES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

st.title("🏦 Loan Default Risk Predictor")
st.caption(
    "An interactive predictive diagnostic system for loan default risk prediction "
    "among underbanked borrowers — XGBoost with cost-sensitive class weighting, "
    "explained with SHAP."
)

st.markdown("---")
st.subheader("Enter Borrower Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    monthly_income = st.number_input("Monthly Income (USD)", min_value=0, value=3000, step=100)
    debt_ratio = st.slider("Debt Ratio", 0.0, 2.0, 0.3, step=0.01)
    revolving_util = st.slider("Revolving Credit Utilization", 0.0, 2.0, 0.3, step=0.01)
    num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)

with col2:
    open_credit_lines = st.number_input("Number of Open Credit Lines/Loans", min_value=0, max_value=50, value=5)
    real_estate_loans = st.number_input("Number of Real Estate Loans/Lines", min_value=0, max_value=20, value=1)
    late_30_59 = st.number_input("Times 30-59 Days Past Due", min_value=0, max_value=20, value=0)
    late_60_89 = st.number_input("Times 60-89 Days Past Due", min_value=0, max_value=20, value=0)
    late_90 = st.number_input("Times 90+ Days Late", min_value=0, max_value=20, value=0)

st.markdown("---")

if st.button("Predict Default Risk", type="primary"):
    try:
        model = load_model()
    except Exception:
        st.error(
            "Model file 'xgboost_model_A.json' not found. Export it from the Colab "
            "notebook (Section 5.2) and place it in this app's folder."
        )
        st.stop()

    explainer = get_explainer(model)

    borrower = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": revolving_util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": num_dependents,
    }])[FEATURE_NAMES]

    proba = model.predict_proba(borrower)[0, 1]
    risk_label = "HIGH RISK" if proba >= 0.5 else "LOW RISK"

    st.subheader("Prediction Result")
    c1, c2 = st.columns(2)
    c1.metric("Default Probability", f"{proba:.1%}")
    c2.metric("Risk Classification", risk_label)

    if proba >= 0.5:
        st.warning(
            "This borrower is flagged as **high risk** of serious delinquency "
            "within 2 years. See the explanation below for contributing factors."
        )
    else:
        st.success(
            "This borrower is flagged as **low risk** of serious delinquency "
            "within 2 years. See the explanation below for contributing factors."
        )

    st.markdown("---")
    st.subheader("Why this prediction? (SHAP Explanation)")
    st.caption(
        "Each bar shows how much a factor pushed the prediction toward higher "
        "risk (red) or lower risk (blue)."
    )

    shap_values = explainer.shap_values(borrower)

    plt.close("all")  # clear any stale figures first
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=borrower.iloc[0].values,
        feature_names=list(borrower.columns),
    )
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()  # grab the actual figure SHAP just drew onto
    fig.set_size_inches(9, 5)
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)

    st.markdown("---")
    st.caption(
        "⚠️ Research/demo interface only — not a deployed production system. "
        "Part of a BSc Software Engineering capstone thesis, ALU."
    )