# IMPLEMENTATION PLAN: AEROFARE (INT395)
**TRANSFORMATION TO PRODUCTION-GRADE ACADEMIC ML APPLICATION**  
**Course:** INT395 — Supervised Learning | Lovely Professional University  
**Developer:** Rithwikkumar35  
**Date:** November 18, 2026  

---

## 1. Plan Overview & Engineering Philosophy

This implementation plan outlines the systematic transformation of the **AeroFare** codebase from a preliminary student script collection into a modular, production-ready, technically defensible, and cloud-deployable machine learning application.

### Guiding Principles:
1. **Zero Destructive Operations**: All original files, original dataset, baseline metrics, and models are backed up and preserved.
2. **Defensible Science**: No fabricated metrics, no fake confidence scores, and no artificial data manipulation.
3. **Deployment Feasibility**: Every artifact committed to the repository must strictly adhere to GitHub's $<100\text{ MB}$ limit and Streamlit Cloud's $\sim 1\text{ GB}$ RAM constraint.
4. **Pedagogical Clarity**: Designed to empower the developer during viva examinations and academic defense.

---

## 2. File Modification & Preservation Strategy

### Files to be Preserved Untouched:
- `data/Clean_Dataset.csv` (The 22.9 MB source dataset remains unaltered).
- `models/best_model.pkl` (Backed up safely to `models/baseline_rf_885mb.pkl` to preserve historical baseline).
- `outputs/airline_average_price.png`, `class_average_price.png`, `days_left_vs_price.png`, `correlation_heatmap.png`, `price_distribution.png` (Existing EDA plots preserved).
- `outputs/airline_class_audit.csv`, `booking_window_audit.csv`, `route_price_audit.csv` (Existing audit tables preserved).
- `app_backup.py` (Preserved as historical reference).

### Files to be Modified / Enhanced:
- `train_model.py`: Enhanced to systematically evaluate baseline algorithms alongside modern tabular models (HistGradientBoosting, ExtraTrees, Regularized Random Forest), computing cross-validation, feature importances, and saving model metadata.
- `model_diagnostics.py`: Extended to calculate granular residual statistics, percentiles, and error distributions by airline, class, and booking window.
- `app.py`: Transformed into a clean, modular entry point orchestrating the multi-page application with cached resources and unified design tokens.

### New Modules & Files to be Created:
- `.gitignore`: Excludes `.venv`, `__pycache__`, `.DS_Store`, and temporary files.
- `.streamlit/config.toml`: Sets aviation-tech theme colors, disables CORS issues, and enforces headless execution.
- `requirements.txt`: Curated, pinned dependency list tested for both macOS and Linux cloud containers.
- `src/`:
  - `src/__init__.py`
  - `src/data_loader.py`: Safe cached dataset and metadata loading with schema validation.
  - `src/preprocessing.py`: Standalone, reproducible preprocessing pipelines and feature definitions.
  - `src/prediction.py`: Robust inference engine with input validation, route verification, and empirical prediction interval calculation.
  - `src/evaluation.py`: Reusable metric calculation, residual analysis, and comparison formatting.
  - `src/utils.py`: Aviation-tech UI components, CSS design tokens, formatting utilities, and disclaimers.
- `pages/`:
  - `pages/1_Home.py`: AeroFare branding, mission, dataset highlights, architecture overview, and quick-start links.
  - `pages/2_Fare_Estimator.py`: Interactive estimation form with physical route validation, interactive inputs, and transparent uncertainty bounds.
  - `pages/3_Model_Performance.py`: Model comparison table, residual plots, feature importance chart, and granular error breakdowns.
  - `pages/4_Dataset_Explorer.py`: High-performance filtered data explorer with distributions, route metrics, and safe 100-record previews.
  - `pages/5_About.py`: Complete methodology breakdown, academic attribution, INT395 course details, and system limitations.
- `models/`:
  - `models/best_model.pkl`: Replaced with an optimized, production-grade model ($<25\text{ MB}$, $R^2 \ge 0.98$) suitable for GitHub and Streamlit Cloud.
  - `models/model_metadata.json`: Feature names, categorical levels, numerical ranges, training date, and scikit-learn version.
- `outputs/`:
  - `outputs/model_comparison.csv`: Updated comparison table including all tested architectures alongside baseline models.
  - `outputs/training_summary.json`: Detailed training hyperparameters, execution time, and evaluation metrics.
  - `outputs/feature_importance.csv`: Feature importances or permutation importances.
- `tests/`:
  - `tests/__init__.py`
  - `tests/test_data.py`: Verifies row count, columns, nulls, and expected types.
  - `tests/test_preprocessing.py`: Tests column transformer isolation and out-of-vocabulary handling.
  - `tests/test_prediction.py`: Tests inference pipeline, identical city rejection, and interval validity.
  - `tests/test_app_smoke.py`: Tests module imports, asset availability, and Streamlit execution readiness.
- Documentation:
  - `TESTING_REPORT.md`: Comprehensive test execution log.
  - `PROJECT_EXPLANATION.md`: In-depth viva conceptual guide covering ML, math, and code.
  - `VIVA_QUESTIONS.md`: 40+ curated examiner Q&As with beginner-friendly explanations.
  - `README.md`: Professional GitHub presentation with badges, screenshots, architecture, and deployment steps.

---

## 3. Phased Execution Roadmap

### Phase 2: ML Improvement & Artifact Optimization
1. Create a safe backup of `models/best_model.pkl` as `models/baseline_rf_885mb.pkl`.
2. Update `train_model.py` to compare:
   - Baseline Linear Regression
   - Baseline Decision Tree Regressor
   - Optimized Random Forest (regularized tree depth $\le 22$ to constrain node growth)
   - Extra Trees Regressor
   - HistGradientBoosting Regressor (modern gradient-boosted decision trees)
3. Select the winning model based on a balanced trade-off between $R^2$/MAE, inference latency, and memory footprint ($<50\text{ MB}$).
4. Serialize the selected model with compression (`joblib.dump(..., compress=3)`).
5. Generate `models/model_metadata.json`, `outputs/training_summary.json`, and `outputs/feature_importance.csv`.
6. Run `model_diagnostics.py` to produce updated class and window performance tables.

### Phase 3: Software Architecture & Modularization
1. Build `src/` modular package:
   - `data_loader.py`: Efficient dataset loading using `@st.cache_data`.
   - `preprocessing.py`: Feature schema, validation rules, column groupings.
   - `prediction.py`: Flight validation logic (e.g. `source != destination`, `duration >= min_flight_time`) and empirical error bands.
   - `evaluation.py`: Real metrics and diagnostic visualizers using Plotly.
   - `utils.py`: Reusable styling cards, badges, and monetary formatters.
2. Ensure relative paths are used throughout (no MacBook hardcoded paths).

### Phase 4: Frontend Redesign (5 Complete Pages)
1. Configure `.streamlit/config.toml` with aviation theme tokens (Navy `#0A192F`, Sky `#0077B6`, Clean Neutral `#F8FAFC`).
2. Construct the 5 pages inside `pages/`:
   - **Page 1 (Home)**: Executive overview, dataset counters, architecture diagram, methodology summary.
   - **Page 2 (Fare Estimator)**: Route selector with instant validation, interactive slider inputs, estimate display, breakdown by component, and prominent academic disclaimer.
   - **Page 3 (Model Performance)**: Interactive Plotly comparisons (MAE, RMSE, $R^2$), actual vs predicted scatter, residual distribution, and feature importance.
   - **Page 4 (Dataset Explorer)**: Route explorer, price distributions, multi-attribute filter with responsive preview table.
   - **Page 5 (About)**: Viva-aligned project explanation, supervised learning formulation, model limitations, and student attribution.
3. Verify responsive layout, contrast ratios, and cross-browser readability.

### Phase 5: Automated Testing & Quality Control
1. Implement `pytest` test suite in `tests/`:
   - Data integrity checks.
   - Preprocessing pipeline tests.
   - Prediction logic and error handling tests.
   - Smoke tests for all app pages.
2. Execute tests in `.venv` and produce `TESTING_REPORT.md`.

### Phase 6: Streamlit Community Cloud Deployment Readiness
1. Generate clean, compatible `requirements.txt`:
   ```
   streamlit>=1.35.0
   pandas>=2.1.0
   numpy>=1.26.0
   scikit-learn>=1.4.0
   joblib>=1.3.0
   plotly>=5.19.0
   ```
2. Verify that total repository size remains well below GitHub limits ($<60\text{ MB}$ total, with dataset $22.9\text{ MB}$ and optimized model $<25\text{ MB}$).
3. Test clean execution using relative paths in a clean subshell.

### Phase 7: GitHub Repository Preparation
1. Create professional `.gitignore` (ignore `.venv`, `__pycache__`, `.DS_Store`, `outputs/*.png` if regenerated, large baseline pickles).
2. Initialize local Git repository: `git init`, set branch `main`.
3. Add remote: `git remote add origin https://github.com/Rithwikkumar35/Airfare-Estimation`.
4. Stage verified files and prepare clean initial commit.
5. **USER APPROVAL GATE**: Present git status and request explicit confirmation before pushing.

### Phase 8: Final Review & Viva Documentation
1. Create `README.md` with:
   - Badges, project overview, and key findings.
   - Model comparison table.
   - Step-by-step Local Setup Guide (macOS / Linux / Windows).
   - Step-by-step Streamlit Cloud Deployment Guide.
   - System limitations and future scope.
2. Create `PROJECT_EXPLANATION.md`: Complete theoretical and practical guide for students.
3. Create `VIVA_QUESTIONS.md`: 40+ comprehensive questions and answers covering supervised regression, algorithms, metrics, data leakage, and deployment.

---

## 4. Immediate Action Item

Phase 1 (Audit & Plan) is complete. We now present this plan to the user for formal approval before initiating Phase 2.
