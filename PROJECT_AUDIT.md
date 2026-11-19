# PROJECT AUDIT: AEROFARE (INT395)
**AIRFARE COMPONENT PRICE ESTIMATION USING SUPERVISED MACHINE LEARNING**  
**Course:** INT395 — Supervised Learning | Lovely Professional University  
**Developer:** Rithwikkumar35  
**Audit Date:** November 18, 2026  
**Auditor:** Senior ML Engineer & Software Architect Review  

---

## 1. Executive Summary

A comprehensive, non-destructive audit of the existing **AeroFare** project located at `/Users/rithwikkumarkodari/Desktop/INT395-Airfare-Estimation` was conducted.

The project demonstrates a solid theoretical foundation for supervised regression. The dataset (`Clean_Dataset.csv`) contains 300,153 verified flight records with zero null values and zero exact duplicate rows. Baseline models (Linear Regression and Random Forest Regressor) were trained, yielding strong initial metrics ($R^2 = 0.9113$ and $R^2 = 0.9849$ respectively).

However, the audit identified **four critical deployment blockers** and several academic weaknesses that must be remediated before publishing to GitHub, deploying to Streamlit Community Cloud, or presenting in an INT395 viva examination:

1. **Catastrophic Model File Size (885.0 MB)**: The serialized Random Forest (`models/best_model.pkl`) is 885 MB because it was trained without depth or leaf node constraints ($12.88\text{ million nodes}$, average depth $42.9$). **GitHub strictly rejects any file larger than 100 MB.** Attempting to push this file will fail immediately. Furthermore, loading an 885 MB pickle on Streamlit Community Cloud causes instant Out-Of-Memory (OOM) container termination.
2. **Missing Deployment Infrastructure**: No `requirements.txt`, no `.gitignore`, no `.streamlit/config.toml`, and no `README.md` exist in the repository.
3. **Uninitialized Git State & Empty Remote**: The local project directory is not yet a Git repository (`fatal: not a git repository`), while the target remote GitHub repository (`https://github.com/Rithwikkumar35/Airfare-Estimation`) exists but is completely empty (zero commits/branches).
4. **Monolithic & Incomplete Frontend Architecture**: `app.py` is an un-modularized 1,544-line file containing only 3 pages (missing the required dedicated Overview/Home and About/Methodology pages), with tightly coupled UI styling, business logic, and prediction execution.

---

## 2. Existing Project Structure & Inventory

```
/Users/rithwikkumarkodari/Desktop/INT395-Airfare-Estimation/
│
├── .venv/                         # Python 3.12.10 virtual environment
├── app.py                         # Monolithic Streamlit application (1,544 lines, 30.9 KB)
├── app_backup.py                  # Early Streamlit backup file (235 lines, 9.4 KB)
├── data/
│   └── Clean_Dataset.csv          # 22.9 MB, 300,153 rows, 11 columns
├── eda.py                         # Exploratory data analysis script (209 lines)
├── fare_audit.py                  # Data quality & price variance audit (165 lines)
├── inspect_data.py                # Dataset inspection script (55 lines)
├── model_diagnostics.py           # Evaluation & error diagnostics script (189 lines)
├── models/
│   └── best_model.pkl             # 885.0 MB uncompressed Random Forest pipeline [CRITICAL BLOCKER]
├── outputs/
│   ├── airline_average_price.png  # EDA plot: Airline vs Price
│   ├── airline_class_audit.csv    # Audit: Class/Airline price summaries
│   ├── booking_window_audit.csv   # Audit: Days left price distribution
│   ├── booking_window_performance.csv # Diagnostic: MAE by booking window
│   ├── class_average_price.png    # EDA plot: Class vs Price
│   ├── class_performance.csv      # Diagnostic: MAE by travel class
│   ├── correlation_heatmap.png    # EDA plot: Numerical correlation matrix
│   ├── days_left_vs_price.png     # EDA plot: Scatter plot (10,000 sampled points)
│   ├── model_comparison.csv       # Baseline evaluation metrics
│   ├── price_distribution.png     # EDA plot: Price histogram
│   ├── route_price_audit.csv      # Audit: Route-level median prices
│   └── test_predictions_diagnostics.csv # 6.8 MB test predictions & residuals
├── preprocess.py                  # Preprocessing demonstration script (126 lines)
└── train_model.py                 # Baseline training script (221 lines)
```

### Missing Standard Files
- `requirements.txt` (Missing — deployment blocker)
- `README.md` (Missing — documentation blocker)
- `.gitignore` (Missing — risk of pushing `.venv`, `.DS_Store`, or raw caches)
- `.streamlit/config.toml` (Missing — Streamlit Cloud server/theme configuration)
- `tests/` directory (Missing — zero automated unit or smoke tests)
- `models/model_metadata.json` (Missing — no serialized inference schema or feature metadata)
- `outputs/training_summary.json` (Missing — no machine-readable training provenance)

---

## 3. Detailed Component Audit

### 3.1 Dataset Quality Audit (`data/Clean_Dataset.csv`)
- **Dimensions**: Exactly 300,153 rows and 11 columns (22.9 MB).
- **Target**: `price` (Continuous integer; range: ₹1,105 to ₹123,071; mean: ₹20,889.66; median: ₹7,425.00).
- **Numerical Features (2)**:
  - `duration`: Continuous float; range: 0.83 to 49.83 hours; mean: 12.22 hours.
  - `days_left`: Integer discrete; range: 1 to 49 days; mean: 26.0 days.
- **Categorical Features (7)**:
  - `airline`: 6 unique categories (`AirAsia`, `Air_India`, `GO_FIRST`, `Indigo`, `SpiceJet`, `Vistara`).
  - `source_city`: 6 unique metro cities (`Bangalore`, `Chennai`, `Delhi`, `Hyderabad`, `Kolkata`, `Mumbai`).
  - `destination_city`: 6 unique metro cities (`Bangalore`, `Chennai`, `Delhi`, `Hyderabad`, `Kolkata`, `Mumbai`).
  - `departure_time`: 6 bins (`Afternoon`, `Early_Morning`, `Evening`, `Late_Night`, `Morning`, `Night`).
  - `arrival_time`: 6 bins (`Afternoon`, `Early_Morning`, `Evening`, `Late_Night`, `Morning`, `Night`).
  - `stops`: 3 categories (`zero`, `one`, `two_or_more`).
  - `class`: 2 categories (`Business`, `Economy`).
- **Identifier**: `flight` (1,561 unique flight codes like `SG-8709`, `UK-995`). Correctly excluded from training.
- **Data Quality Assessment**:
  - Missing values: **0** across all 11 columns.
  - Exact duplicate rows: **0**.
  - Extreme values: Price has a heavy right tail (skewness $\approx 1.06$). High prices are almost exclusively Business class tickets (median ₹53,164 vs ₹5,772 for Economy). These are **valid real-world fares** and must not be trimmed as statistical outliers.

### 3.2 Preprocessing Audit (`preprocess.py` & Pipeline)
- **Train/Test Isolation**: Preprocessing uses `ColumnTransformer` and fits solely on `X_train` ($80\%$ split, 240,122 rows), transforming `X_test` ($20\%$ split, 60,031 rows). There is no feature leakage during transformation.
- **Encoders**:
  - `StandardScaler` on numerical features (`duration`, `days_left`).
  - `OneHotEncoder(handle_unknown="ignore")` on categorical features.
- **Weakness**:
  - Target variable (`price`) was not log-transformed. While tree models are invariant to monotonic transformations of the target, linear regression performs sub-optimally with un-transformed skewed targets.
  - The feature column order and expected category sets are not saved as a standalone JSON artifact, forcing inference to rely on implicit pipeline states.

### 3.3 Model Training & Serialization Audit (`train_model.py` & `models/best_model.pkl`)
- **Baseline Models Tested**:
  1. `LinearRegression()`
  2. `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_features=1.0)`
- **The 885 MB Random Forest Problem**:
  - Inspection of `models/best_model.pkl` revealed:
    - Estimators: 100 decision trees.
    - Total tree nodes: **12,881,684 nodes**.
    - Average tree depth: **42.91**.
    - `max_depth`: `None` (unbounded).
    - `min_samples_split`: `2`.
    - `min_samples_leaf`: `1`.
    - Compression: None (raw Python pickle).
  - **Impact**:
    - Serialized file size: **885.0 MB**.
    - With `joblib.dump(compress=3)`: **175.5 MB** (still exceeds GitHub's 100 MB hard limit).
    - GitHub will reject `git push origin main`.
    - Streamlit Cloud free tier containers have ~1 GB RAM. Loading this model causes immediate crash.
- **Empirical Benchmarks Completed During Audit**:
  - **Pruned Random Forest** (`n_estimators=100, max_depth=22, min_samples_leaf=2`, compressed):
    - Size: **90.67 MB** (under GitHub 100 MB limit).
    - Test $R^2$: **0.9856** (higher than baseline 0.9849!).
    - Test RMSE: **₹2,722.18** (better than baseline ₹2,787.82!).
    - Test MAE: **₹1,201.12**.
  - **HistGradientBoostingRegressor** (`max_iter=200`, native histogram binning):
    - Size: **0.29 MB** (290 KB!).
    - Training time: **4.07 seconds** (vs 72s for RF).
    - Test $R^2$: **0.9744**.
    - Test RMSE: **₹3,632.66**.
    - Test MAE: **₹2,125.68**.
    - Memory load: Instant, zero latency, ultra-lightweight for cloud deployment.

### 3.4 Baseline Metrics Verification

| Model | Baseline Claimed | Verified in `outputs/model_comparison.csv` | Status |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | MAE: ₹4,553.29<br>RMSE: ₹6,761.71<br>$R^2$: 0.911305 | MAE: ₹4,553.29<br>RMSE: ₹6,761.71<br>$R^2$: 0.911305 | **Exact Match (100% verified)** |
| **Random Forest** | MAE: ₹1,077.91<br>RMSE: ₹2,787.82<br>$R^2$: 0.984923 | MAE: ₹1,077.91<br>RMSE: ₹2,787.82<br>$R^2$: 0.984923 | **Exact Match (100% verified)** |
| **Business Class MAE** | ₹2,142.30 | ₹2,142.30 (`class_performance.csv`) | **Exact Match** |
| **Economy Class MAE** | ₹595.77 | ₹595.77 (`class_performance.csv`) | **Exact Match** |
| **1–7 Days Left MAE** | ₹2,624.50 | ₹2,624.50 (`booking_window_performance.csv`) | **Exact Match** |
| **41–49 Days Left MAE**| ₹680.70 | ₹680.70 (`booking_window_performance.csv`) | **Exact Match** |

All baseline numbers are authentic and reproducible from existing project files.

### 3.5 Frontend Application Audit (`app.py`)
- **Structure**: Single monolithic file (1,544 lines).
- **Navigation**: Controlled by a single sidebar `st.radio` with only 3 options:
  1. `✈️ Fare Estimator`
  2. `📊 Model Insights`
  3. `🗂️ Dataset Explorer`
- **Missing Pages**:
  - Page 1: Home / Overview (Branding, key stats, project objectives, estimation explanation).
  - Page 5: About / Methodology (Supervised learning workflow, viva defense, limitations, tech stack).
- **Design & UI Weaknesses**:
  - Heavy inline CSS with explicit light backgrounds (`#f7faff`, `#ffffff`) combined with dark text variables. In dark-themed browsers, custom styles can clash with Streamlit defaults.
  - Raw HTML injection inside Streamlit containers (`unsafe_allow_html=True`) without standard component fallback.
  - No input validation preventing identical Source and Destination cities (e.g., Mumbai $\rightarrow$ Mumbai).
  - No uncertainty explanation on predictions.

### 3.6 Git and Remote Status Audit
- **Local Git Repository**: `None`. No `.git` folder exists in `/Users/rithwikkumarkodari/Desktop/INT395-Airfare-Estimation`.
- **Remote Repository**: `https://github.com/Rithwikkumar35/Airfare-Estimation`
  - Checked via `git ls-remote`: **0 refs, completely empty**.
  - Confirms user statement: The remote repository is ready for its clean initial push.

---

## 4. The 7 Required Identification Points

### 1. What currently works
- Complete, clean dataset with 300,153 validated records.
- Standard scikit-learn preprocessing pipeline with `ColumnTransformer`.
- Baseline Linear Regression and Random Forest training and evaluation scripts.
- Diagnostic analysis scripts generating verified metrics by class, airline, and booking window.
- Core inference calculation in `app.py` successfully loading model and producing price estimates.

### 2. What is incomplete
- Multi-page application structure: currently only 3 pages exist; 5 cohesive pages are required.
- No `requirements.txt` file specifying verified library versions.
- No comprehensive academic `README.md` with visual architecture, installation, and deployment instructions.
- No automated test suite (`tests/` for unit, data integrity, pipeline, and app smoke tests).
- No inference schema metadata JSON file (`models/model_metadata.json`).
- No machine-readable training log (`outputs/training_summary.json`).

### 3. What is incorrect
- Model persistence without size or depth management: saving a 100-tree unpruned Random Forest resulting in an 885 MB file.
- The UI allows selecting the same city for departure and arrival (e.g., Mumbai $\rightarrow$ Mumbai), which is physically invalid for commercial flights.
- UI does not gracefully validate flight duration against physical route minimums (e.g., Delhi to Mumbai cannot be 0.1 hours).

### 4. What can be improved
- **Model Efficiency & Diversity**: Evaluate and compare `HistGradientBoostingRegressor` and regularized tree ensembles that offer comparable accuracy ($R^2 \approx 0.98$) at $<1\%$ of the file size ($0.29\text{ MB} - 25\text{ MB}$).
- **Modular Software Architecture**: Extract logic into `src/` modules (`data_loader.py`, `preprocessing.py`, `prediction.py`, `evaluation.py`, `utils.py`).
- **UI/UX Design**: Implement a cohesive aviation-grade design system (deep navy, aviation blue, soft neutrals) with Streamlit native cards, metrics, and Plotly interactive visualizations.
- **Statistical Uncertainty**: Provide defensible prediction intervals based on observed test-set residual distributions by class rather than fabricated percentages.

### 5. What should not be changed / preserved
- `data/Clean_Dataset.csv`: Must remain untouched in its original location.
- Original baseline metrics: Must be preserved in documentation and comparative tables.
- Original backup file `app_backup.py` and original scripts: Preserved.
- The existing `models/best_model.pkl`: Preserved in a backup directory (`models/archive/` or `models/baseline_rf_885mb.pkl`) to guarantee zero loss of existing work.

### 6. What will break during deployment (Critical Blockers)
1. **GitHub 100 MB Limit**: Committing or pushing `models/best_model.pkl` (885 MB) will cause GitHub to reject the push with a fatal hook error.
2. **Streamlit Cloud RAM Limit**: Loading an 885 MB pickle will crash the Streamlit Community Cloud runner ($1\text{ GB} - 2.7\text{ GB}$ ceiling) with an Out-of-Memory (OOM) signal.
3. **Missing `requirements.txt`**: Streamlit Cloud will fail on build because it cannot identify dependencies.
4. **Git Repository Uninitialized**: The project directory cannot be pushed until `git init`, `.gitignore`, and the initial commit are properly configured.

### 7. What is technically weak or difficult to defend in a viva
- **Examiner Question**: *"Why is your saved model 885 MB for a 23 MB dataset?"*  
  **Weakness**: Reveals lack of hyperparameter tuning, failure to regularize tree depth, and lack of knowledge about tree node growth in scikit-learn.
- **Examiner Question**: *"Did you test for data leakage across flight numbers and dates?"*  
  **Weakness**: Random 80:20 splitting places identical flight numbers operating on adjacent days into both train and test splits, causing optimistic test metrics.
- **Examiner Question**: *"Why did you not evaluate modern gradient boosting algorithms?"*  
  **Weakness**: Modern tabular ML relies heavily on Gradient Boosted Decision Trees (HistGradientBoosting, LightGBM, XGBoost) rather than unconstrained Random Forests.
- **Examiner Question**: *"Is this model suitable for real-time airline ticket pricing?"*  
  **Weakness**: Fails to distinguish between static component estimation from a historical snapshot and dynamic revenue management algorithms with real-time seat inventory.

---

## 5. Audit Conclusion & Next Steps

The codebase is structurally sound and mathematically valid, but requires professional refactoring for deployment readiness. 

Proceed to review `IMPLEMENTATION_PLAN.md` for the step-by-step remediation plan across Phases 2 through 8.
