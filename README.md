# Electric Vehicle Battery Performance and Range Prediction

An end-to-end Machine Learning and Data Visualization engineering project developed to analyze electric vehicle battery telemetry, benchmark regression models, and deliver real-time driving range predictions via an interactive four-page Streamlit application.

---

## 1. Project Overview

Range anxiety remains one of the primary obstacles to widespread electric vehicle (EV) adoption. Accurately estimating remaining driving range requires modeling complex interactions between electrochemical battery conditions, vehicle dynamics, road terrain, and environmental parameters. This project implements a complete data science lifecycle—from exploratory data analysis and feature preprocessing to multi-model benchmarking, diagnostic validation, and dashboard deployment.

---

## 2. Problem Statement

Estimating EV remaining driving range is challenging because energy consumption fluctuates dynamically depending on speed, terrain slope, battery health, and accessory loads. Fixed-average estimation methods fail to account for non-linear operating thresholds and operating resistance, leading to either over-optimistic range estimates or premature battery depletion. The objective is to evaluate regression architectures capable of capturing these relationships and deploy the best-performing pipeline into an accessible user interface.

---

## 3. Project Objectives

1. **Verify Data Integrity:** Audit dataset dimensions, distributions, missing values, and domain boundaries.
2. **Feature Engineering & Leakage Isolation:** Build modular preprocessing pipelines ensuring complete isolation between training and holdout test data.
3. **Model Benchmarking:** Train and compare diverse regression architectures (Mean Baseline, Linear Regression, K-Nearest Neighbors, and Random Forest) using a chronological 80:20 evaluation split.
4. **Diagnostic Error Analysis:** Investigate residual patterns, error distributions, and feature importances.
5. **Interactive Deployment:** Deliver a responsive four-page Streamlit dashboard supporting real-time range inference and diagnostic exploration.

---

## 4. Dataset Description & Synthetic Data Disclosure

* **File Location:** `data/EV_Range_Prediction_Dataset_2000.csv`
* **Dataset Shape:** 2,000 records, 17 columns (1 timestamp, 15 predictor features, 1 target).

### Technical Provenance & Mathematical Formula Disclosure:
- **Synthetic Ground Truth:** Empirical audit demonstrates that the dataset is synthetically generated rather than recorded from real-world EV CAN-bus telemetry.
- **Discovered Generator Function:** Across all 2,000 records, the target variable `remaining_range` matches an exact deterministic piecewise-linear equation (to within $\le 0.005\text{ km}$ numerical rounding):
  $$\text{remaining\_range} = \max\left(5.0,\, 3.0 \times \text{battery\_soc} + 1.8 \times \text{battery\_health} - 2.0 \times \text{road\_gradient} - 1.5 \times \text{power\_consumption} - 0.2 \times \text{speed} - 180.0\right)$$
- **Boundary Clamping:** Exactly **312 rows (15.6%)** are clamped at the lower bound floor of **$5.00\text{ km}$**. No values exist below $5.00\text{ km}$.
- **Zero-Weight Noise Predictors:** Exactly **10 of the 15 features** (`temperature`, `humidity`, `wind_speed`, `voltage`, `current`, `vehicle_load`, `regen_braking`, `distance_traveled`, `traffic_density`, and `driving_style`) have zero mathematical weight in the target equation.
- **Observation Independence:** Consecutive 1-minute timestamps exhibit instantaneous physical jumps (e.g., SOC swinging by $>50\%$ in 60 seconds), confirming observations are independent samples rather than continuous drive cycles.

---

## 5. Features and Technology Stack

### Features (15 Predictors + 1 Target):
* **Battery Information:** `battery_soc` (%), `battery_health` (%), `voltage` (V)
* **Electrical & Vehicular:** `current` (A), `power_consumption` (kW), `speed` (km/h), `distance_traveled` (km), `road_gradient` (%), `vehicle_load` (kg), `regen_braking` (%)
* **Environmental Conditions:** `temperature` (°C), `humidity` (%), `wind_speed` (m/s)
* **Driving Behaviour:** `traffic_density` (Low, Medium, High), `driving_style` (Eco, Normal, Aggressive)
* **Target Variable:** `remaining_range` (km)

### Technology Stack:
* **Core Language:** Python 3.10+ (tested on Python 3.15)
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Scientific Computing:** SciPy
* **Visualization:** Plotly Express, Matplotlib, Seaborn
* **Model Serialization:** Joblib
* **Application Framework:** Streamlit

---

## 6. EDA and Preprocessing Workflow

1. **Quality Checks:** Confirmed 0 missing cells across all 15 predictors and 0 missing target entries. Verified domain boundaries.
2. **Modular ColumnTransformer Pipeline:**
   - **Numeric Pipeline:** Median imputation via `SimpleImputer(strategy="median")`. Standardized with `StandardScaler()` for Linear Regression and KNN; left in native scale for tree-based estimators.
   - **Categorical Pipeline:** Mode imputation via `SimpleImputer(strategy="most_frequent")` followed by `OneHotEncoder(handle_unknown="ignore")`.
   - **Remainder:** `remainder="drop"` to exclude non-feature columns (e.g., `timestamp`).
3. **Leakage Prevention:** The entire preprocessing pipeline is fitted strictly on the training subset ($N=1,600$), preventing information leakage into the test set ($N=400$).

---

## 7. Machine Learning Models Evaluated

Four regression models were implemented and evaluated within full scikit-learn pipelines:
1. **Mean Baseline:** `DummyRegressor(strategy="mean")`
2. **Linear Regression:** Ordinary least squares with standardized features.
3. **K-Nearest Neighbors (KNN):** `KNeighborsRegressor(n_neighbors=7, weights="distance")` with standardized features.
4. **Random Forest Regressor:** `RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)` without feature scaling.

---

## 8. Final Verified Model Metrics

Evaluation on the 400-record holdout test split yielded the following verified performance:

| Model | MAE (km) | RMSE (km) | MSE ($\text{km}^2$) | $\text{R}^2$ | Evaluation Split |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **6.3057** | **8.2731** | **68.4446** | **0.98804** | Chronological 80/20 |
| **Linear Regression** | **6.1162** | **8.9744** | **80.5404** | **0.98592** | Chronological 80/20 |
| **KNN Regression** | 27.6210 | 33.2937 | 1108.4687 | 0.80623 | Chronological 80/20 |
| **Mean Baseline** | 65.6441 | 75.6336 | 5720.4366 | -0.00000 | Chronological 80/20 |

**Selected Model:** The **Random Forest Regressor** is the chosen production pipeline because it achieves the lowest Root Mean Squared Error (**8.2731 km**) and highest $R^2$ (**0.98804**), accurately capturing the non-linear boundary clamp at 5 km.

---

## 9. Model Limitations

1. **Synthetic Scope:** High performance ($R^2 = 0.988$) demonstrates successful functional recovery of the underlying generator formula, not empirical validation on physical EV battery cells.
2. **Drive Cycle Discontinuity:** Because observations are independent, the model does not capture state-of-charge time-series dynamics (e.g., Kalman filtering or LSTM-based battery state estimation).
3. **Clamping Bias:** Linear models produce larger tail penalties due to underpredicting at the artificial $5.00\text{ km}$ boundary floor.

---

## 10. Four-Page Dashboard Description

The Streamlit web application (`dashboard/app.py`) is organized into four clean views with a horizontal top navigation bar:

1. **Home / Overview:** Project title, four dynamic KPI summary cards (Total Records, Avg SOC, Avg Health, Avg Range), application purpose, research findings, and navigation guide.
2. **Range Predictor:** Interactive 15-feature input form with dataset-derived slider/number bounds, calling the saved Random Forest pipeline to generate real-time range predictions and battery status.
3. **Battery Analytics:** Telemetry distribution histogram, SOC vs. Range scatterplot, and full Prediction Diagnostics (Actual vs. Predicted with $y=x$ reference line, Residuals vs. Predicted with $y=0$ line, absolute error distribution, and test set summary cards).
4. **Model Insights:** Benchmark comparison table, comparative RMSE bar chart highlighting the top model, top-10 feature importance horizontal bar chart, and dynamic model architecture metadata.

---

## 11. Project Folder Structure

```
DashBoard/
├── data/
│   └── EV_Range_Prediction_Dataset_2000.csv    # Project dataset (2,000 records)
├── src/
│   ├── check_data.py                           # Dataset integrity verification script
│   ├── review3_ml_analysis.py                  # Core ML training, evaluation & export script
│   ├── test_saved_model.py                     # Joblib model loading and inference test
│   ├── best_model_pipeline.joblib              # Serialized Random Forest pipeline
│   ├── best_model_predictions.csv              # Test set predictions (actual vs predicted)
│   ├── model_metrics.csv                       # Verified model comparison metrics
│   ├── random_forest_feature_importance.csv    # Feature importances
│   ├── cleaned_range_data.csv                  # Cleaned dataset export
│   ├── data_profile.csv                        # Column data profiles
│   ├── descriptive_statistics.csv              # Feature summary statistics
│   ├── domain_range_checks.csv                 # Boundary range validations
│   ├── normality_checks.csv                    # Normality diagnostic outputs
│   ├── outlier_report.csv                      # IQR outlier report
│   ├── run_summary.json                        # Comprehensive run metadata
│   └── figures/                                # Generated static diagnostic plots
├── dashboard/
│   └── app.py                                  # Four-page interactive Streamlit dashboard
├── outputs/
│   ├── results_and_findings.md                 # Technical review findings and report
│   ├── EV Energy Efficiency Dataset.csv        # Reference NRCAN EV efficiency dataset
│   └── figures/                                # Reference review plots
├── .gitignore                                  # Git exclusion configuration
├── requirements.txt                            # Python package dependencies
└── README.md                                   # Comprehensive project documentation
```

---

## 12. Installation Instructions

### Prerequisites:
* Python 3.10, 3.11, 3.12, or 3.15 installed on your machine.
* PowerShell or Command Prompt (Windows), or Bash (Linux/macOS).

### Setup Virtual Environment:
```powershell
# Navigate to the project root directory
cd "d:\RAVI\PROJECTS\Software's\EV Prediction\DashBoard"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install required dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 13. How to Run the ML Analysis Script

To execute the data preprocessing, model training, evaluation, and pipeline export:

```powershell
python src/review3_ml_analysis.py
```
*Outputs generated:* `best_model_pipeline.joblib`, `model_metrics.csv`, `best_model_predictions.csv`, `random_forest_feature_importance.csv`, and diagnostic figures in `src/`.

To run model verification independently:
```powershell
python src/test_saved_model.py
```

---

## 14. How to Launch the Streamlit Dashboard

To start the interactive four-page web application:

```powershell
streamlit run dashboard/app.py
```

The application will open automatically in your default browser at `http://localhost:8501`.

---

## 15. Project Team

* **Vishwakarma Ravi Kiran** – `2320040081`
* **Peddigari Chetan** – `2320040067`
* **Prajwal Auradkar** – `2320040084`

**Department of Electronics and Communication Engineering (ECE)**  
**Data Science and Visualization**  
**Academic Year 2026–2027**  

