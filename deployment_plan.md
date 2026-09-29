# Deployment Plan

**Project:** Loan Default Risk Prediction & Explanation Stability Audit
**Author:** Henriette Utatsineza 

## Current Stage: Local Research Demo (this submission)

At this stage, the system is deployed as a **local, single-user demonstration** rather than a hosted service, consistent with the project's defined scope (thesis Section 1.5 — an offline, dataset-based research investigation rather than a live deployment).

**What is deployed now:**
- A Streamlit web interface (`app.py`) that runs on `localhost` via `streamlit run app.py`
- The interface loads a pre-trained XGBoost model (`xgboost_model_A.json`, exported from the training notebook) and a SHAP `TreeExplainer` built on that model
- A user enters one borrower's financial details through a form and receives, in real time: (1) a default-risk probability, (2) a risk classification, and (3) a SHAP waterfall explanation of the contributing factors

**Why local-only at this stage:**
- The project's core research contribution — the SHAP explanation stability audit — is an offline analysis (comparing two model variants on a fixed sample of borrowers), not a live user-facing feature
- No real borrower or institutional data is involved; the public Kaggle dataset does not require secure hosting or access controls
- A three-month capstone timeline prioritizes the modeling and audit methodology over deployment infrastructure

## Architecture (Current)

```
[Browser] <--> [Streamlit app, local process]
                     |
                     v
          [XGBoost model (xgboost_model_A.json)]
                     |
                     v
          [SHAP TreeExplainer]
```

No database, no API layer, no authentication — a single local process serving one user at a time, matching the "lightweight local demonstration interface" described in the project's system architecture (thesis Section 3.4).

## Future Deployment Path (Beyond This Submission)

If this project were extended toward real-world use by a lending institution, the following stages would be the realistic next steps — noted here as planned future work, not part of the current deliverable:

1. **Containerization** — package the Streamlit app and model artifact in a Docker container for consistent, reproducible deployment across environments.
2. **Cloud hosting** — deploy via a platform such as Streamlit Community Cloud (free tier, suitable for a public demo link) or a cloud provider (AWS/GCP) for a more production-oriented setup.
3. **API layer** — expose the model via a REST API (e.g. FastAPI) so the prediction and explanation logic could be integrated into an existing loan-origination system rather than used only through a standalone form.
4. **Access control and audit logging** — required before any real borrower data could be processed, to meet data protection and financial-sector regulatory expectations.
5. **Model versioning and monitoring** — track model drift over time and re-run the SHAP stability audit periodically as the model is retrained on new data, since this project's own findings suggest explanation stability cannot be assumed to persist indefinitely.

## Rollback / Reliability Notes (Current Local Setup)

- The trained model and all intermediate artifacts (SHAP values, train/test split, performance metrics) are version-controlled by saving to Google Drive at each pipeline stage (see notebook Section 1.1), so a corrupted or lost local environment does not require retraining from scratch.
- Since the app loads a static, pre-exported model file rather than retraining on each run, restarting the Streamlit process has no effect on prediction consistency.