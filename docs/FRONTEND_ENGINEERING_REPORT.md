# Frontend Engineering Report

## 1. Repository State Before My Changes

When I started, the repository was a machine learning proof-of-concept rather than a demoable application. The codebase contained:

- A project README describing the business problem and intent
- Architecture notes in `docs/HLD.md` and `docs/LLD.md`
- Three notebook-style Python scripts under `.py files/`
- A processed dataset at `data/processed/final_poc_dataset.zip`

What it did not contain was equally important:

- No saved trained model artifact
- No reproducible training entry point
- No reusable inference helper
- No frontend
- No API
- No dependency manifest

In short, the repository had the baseline ingredients for modeling, but it was not yet in a state that could be demonstrated live.

## 2. Problems That Prevented a Demo

Several concrete issues blocked a checkpoint presentation:

- The baseline model was only trained in-script and never persisted for reuse
- There was no stable inference entry point for a frontend to call
- The repository had no `requirements.txt`, so setup was not reproducible
- The existing scripts were oriented toward experimentation, not demo execution
- There was no UI for entering inputs, loading scenarios, or showing outputs
- The README did not reflect the real runnable workflow

Any live demo would have required ad hoc setup and manual code editing, which was too risky for a presentation.

## 3. What I Implemented

My contribution focused on making the repository demoable with the smallest practical change set, while keeping the original project intent and the existing logistic regression baseline intact.

I implemented:

- A reproducible baseline training script that reads the committed processed dataset
- A saved sklearn model artifact for demo reuse
- A small shared inference and feature-contract module
- A thin Streamlit frontend for checkpoint presentation use
- One-click demo scenarios to make the presentation reliable
- A README update aligned to the actual implemented workflow
- A startup safety check so the frontend verifies that the saved model artifact can actually be loaded before allowing predictions

I did not redesign the model, introduce a new backend architecture, or add fake production claims.

## 4. Files Created And Modified

### Created

- `docs/superpowers/specs/2026-04-17-demo-baseline-design.md`
- `requirements.txt`
- `model_contract.py`
- `train_baseline_model.py`
- `artifacts/baseline_logreg_pipeline.joblib`
- `streamlit_app.py`

### Modified

- `README.md`
- `streamlit_app.py` after initial review, to validate model loading at startup

### This Report

- `docs/FRONTEND_ENGINEERING_REPORT.md`

## 5. Why Streamlit Was Chosen

Streamlit was chosen because it was the fastest safe path to a presentation-ready frontend by the checkpoint deadline.

It fit the problem well for four reasons:

- The model already ran in Python, so no API layer was required
- The UI needed only a small number of numeric controls and clear outputs
- Demo scenarios could be added quickly and safely
- The implementation cost was low enough to keep focus on reliability rather than framework overhead

For this checkpoint, Streamlit provided the shortest path from trained artifact to usable interface.

## 6. How The Frontend Connects To The Trained Model

The frontend does not retrain the model.

Instead, it uses the saved artifact at:

- `artifacts/baseline_logreg_pipeline.joblib`

The integration path is:

1. `streamlit_app.py` collects the 7 baseline input features
2. The app calls `predict_delay(features)` from `model_contract.py`
3. `model_contract.py` loads the saved sklearn pipeline artifact
4. The helper builds a one-row dataframe in the exact expected feature order
5. The pipeline returns both predicted class and delay probability
6. The frontend maps the probability to a presentation-friendly risk band and recommended business action

This keeps the frontend thin and aligned to the saved baseline model.

## 7. Demo Scenarios Added

To make the live presentation more reliable, I added five one-click demo scenarios:

- `Routine order`
- `Borderline watchlist`
- `Approval bottleneck`
- `Weekend escalation`
- `Severe friction`

These scenarios were chosen to produce visibly different outputs, including low-, medium-, and high-risk style examples for presentation flow.

## 8. How To Run The Demo

### Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Train and save the baseline artifact

```bash
python3 train_baseline_model.py
```

### Run the frontend

```bash
python3 -m streamlit run streamlit_app.py
```

## 9. Current Limitations

The repository is now demoable, but it is still a checkpoint build rather than a production system.

Current limitations include:

- The frontend is local-first and depends on the committed model artifact
- There is no API or deployment layer
- The baseline model remains the original logistic regression approach
- Input validation is structural, not deeply business-semantic
- Predictions are suitable for demo decision support, not operational automation
- The UI is intentionally minimal and does not include analytics, history, authentication, or audit features

## 10. What I Would Improve Next With More Time

If given more time, I would improve the solution in the following order:

1. Add lightweight automated tests for the training path, artifact loading, and frontend prediction flow
2. Add stronger business-level validation and explanatory guidance around feature inputs
3. Introduce model caching in the frontend to avoid repeated artifact loads during a session
4. Add a small API boundary so the UI and inference logic are more cleanly separated
5. Improve demo observability with clearer input summaries, confidence context, and prediction explanation cues
6. Replace the checkpoint-style setup with a more complete packaging and environment workflow

## Summary

My contribution was to turn an ML proof-of-concept repository into a demo-ready checkpoint build with minimal engineering disruption.

I kept the original project direction intact, reused the existing baseline model, and added the smallest practical training, artifact, inference, frontend, and documentation pieces needed for a reliable live presentation.
