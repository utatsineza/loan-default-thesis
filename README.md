# Loan Default Risk Prediction — Initial Software Demonstration

**Project:** An Interactive Predictive Diagnostic System for Loan Default Risk Prediction Among Underbanked Borrowers Utilizing XGBoost with Cost-Sensitive Class Weighting and a SHAP-Based Explanation Stability Audit

**Author:** Henriette Utatsineza 
**Track:** ML Track — BSc (Hons) Software Engineering, African Leadership University
**Supervisor:** Elvira Khwatenge

---

## Description

This project predicts the probability that a borrower will experience serious loan default (90+ days delinquent) within two years, using the public **Give Me Some Credit** dataset (Kaggle, ~150,000 borrower records). Class imbalance (~6.7% default rate) is addressed through **cost-sensitive class weighting** in XGBoost (`scale_pos_weight`) rather than synthetic resampling (e.g. SMOTE), preserving the true feature-space distribution that SHAP's explanations depend on.

Every prediction is paired with a **SHAP explanation** showing exactly which borrower factors drove the risk score up or down — this project's broader research goal (beyond this initial demo) is to audit whether two independently trained, equally accurate models agree on these explanations for the same borrower.

This submission covers the **initial software demonstration**:
- A model notebook (data visualization, preprocessing, baseline + XGBoost training, initial performance metrics, first SHAP explanation)
- A lightweight local deployment mockup (Streamlit web interface) demonstrating the prediction + explanation flow end-to-end

## Link to GitHub Repo

`[INSERT YOUR GITHUB REPO URL HERE]`

## How to Set Up the Environment and the Project

### Option A — Run the Notebook (Google Colab)
1. Open `loan_default_initial_demo.ipynb` in Google Colab.
2. Download `cs-training.csv` from [Kaggle: Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) and upload it to the Colab session (folder icon → upload).
3. Run all cells in order (Runtime → Run all). The first Drive-mounting cell will prompt a one-time authorization popup.
4. All models, plots, and results auto-save to `MyDrive/loan-default-thesis/` as the notebook runs.

### Option B — Run the Deployment Mockup (Streamlit, local)
1. Clone this repo and `cd` into it.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Export the trained model from the notebook (already handled automatically in Section 5.2 — it saves as `xgboost_model_A.json`) and place that file in the same folder as `app.py`.
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. Open the local URL Streamlit prints (typically `http://localhost:8501`), enter a borrower's details, and click **Predict Default Risk**.

## Designs

See `/screenshots` for:
- The borrower input form
- A "LOW RISK" prediction result
- The SHAP waterfall explanation for that prediction

*(Add your actual screenshot files to a `/screenshots` folder in the repo, matching the ones used in your video demo.)*

## Deployment Plan

See [`DEPLOYMENT_PLAN.md`](./DEPLOYMENT_PLAN.md) for the full write-up.

## Repository Structure

```
├── README.md
├── DEPLOYMENT_PLAN.md
├── loan_default_initial_demo.ipynb
├── app.py
├── requirements.txt
├── xgboost_model_A.json          (exported from the notebook — add after running it)
└── screenshots/
    ├── 01_input_form.png
    ├── 02_prediction_result.png
    └── 03_shap_explanation.png
```

## Tech Stack

- **Language:** Python 3
- **Data handling:** pandas, NumPy
- **Modeling:** XGBoost, scikit-learn
- **Explainability:** SHAP (TreeExplainer)
- **Interface:** Streamlit
- **Development environment:** Google Colab / Jupyter Notebook

## Note on Scope

This is a research/demo interface, not a deployed production system. Per the project's defined scope (see thesis Section 1.5), the system is evaluated entirely offline on public data; it is not integrated with any live lending institution's systems.