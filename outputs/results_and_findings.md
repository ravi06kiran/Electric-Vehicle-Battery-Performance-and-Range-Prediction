# Review 3 Results and Technical Findings
**Project:** Electric Vehicle Battery Performance and Range Prediction  
**Department:** Department of Electronics and Communication Engineering (ECE)  
**Course:** Data Science and Visualization | Academic Year 2026–2027  

---

## 1. Dataset Origin & Synthetic Data Disclosure

The analysis utilizes `EV_Range_Prediction_Dataset_2000.csv` containing 2,000 observations and 17 columns (1 timestamp, 15 predictor features, and 1 continuous target: `remaining_range` in km).

### Critical Data Provenance & Formula Disclosure:
- **Synthetic Ground Truth:** The dataset is strongly evidenced to be synthetically generated rather than real-world EV CAN-bus or telemetry logs.
- **Discovered Target Equation:** Rigorous empirical regression across all 2,000 records proves that `remaining_range` follows an exact deterministic piecewise-linear mathematical formula:
  $$\text{remaining\_range} = \max\left(5.0,\, 3.0 \times \text{battery\_soc} + 1.8 \times \text{battery\_health} - 2.0 \times \text{road\_gradient} - 1.5 \times \text{power\_consumption} - 0.2 \times \text{speed} - 180.0\right)$$
- **Boundary Clamping:** Exactly **312 records (15.60%)** are clamped at the lower bound floor of **$5.00\text{ km}$**. Zero observations fall below $5.00\text{ km}$.
- **Zero-Weight Predictors:** Exactly **10 of the 15 predictor variables** (`temperature`, `humidity`, `wind_speed`, `voltage`, `current`, `vehicle_load`, `regen_braking`, `distance_traveled`, `traffic_density`, and `driving_style`) have **zero mathematical influence** on the target variable. They act as independent noise features.
- **Independence & Temporal Nature:** Consecutive 1-minute timestamps exhibit physically impossible instantaneous jumps (e.g., SOC swinging by $>50\%$ in 60 seconds). Therefore, observations are independently generated points rather than a continuous vehicular driving cycle.
- **Split Qualification:** The 80:20 chronological partition (first 1,600 rows train, final 400 rows test) does **not** represent genuine temporal out-of-sample forecasting; it functions essentially as an arbitrary holdout split on synthetic data.
- **Real-World Scope:** The high $R^2$ scores ($>0.98$) reflect successful identification of the underlying piecewise formula and do **not** prove real-world EV prediction accuracy under physical battery degradation and unobserved operating conditions.

---

## 2. Data Quality & Preprocessing

- **Missing Values:** 0 missing values across all 15 predictor columns; 0 missing target rows.
- **Domain Sanity Checks:** 0 out-of-range observations across SOC ($0\text{--}100\%$), health ($0\text{--}100\%$), humidity ($0\text{--}100\%$), speed ($\ge 0$), and distance traveled ($\ge 0$).
- **Outlier Handling:** Interquartile Range (IQR) checks flagged extreme operating points; all rows were retained to capture full domain variation.
- **Preprocessing Pipeline:**
  - Implemented via scikit-learn `ColumnTransformer`.
  - Numeric features: Median imputation (`SimpleImputer(strategy="median")`); standardized (`StandardScaler()`) for linear and distance-based models; kept unscaled for tree-based models.
  - Categorical features (`traffic_density`, `driving_style`): Mode imputation (`SimpleImputer(strategy="most_frequent")`) and one-hot encoding (`OneHotEncoder(handle_unknown="ignore")`).
  - Preprocessor was fitted strictly on training data ($N=1,600$) to guarantee zero test leakage.

---

## 3. Verified Model Evaluation & Comparison

All models were evaluated on the identical 400-record holdout test split. Below are the verified final metrics:

| Model | MAE (km) | RMSE (km) | MSE ($\text{km}^2$) | $\text{R}^2$ | Split |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **6.3057** | **8.2731** | **68.4446** | **0.98804** | Chronological 80/20 |
| **Linear Regression** | **6.1162** | **8.9744** | **80.5404** | **0.98592** | Chronological 80/20 |
| **KNN Regression** | 27.6210 | 33.2937 | 1108.4687 | 0.80623 | Chronological 80/20 |
| **Mean Baseline** | 65.6441 | 75.6336 | 5720.4366 | -0.00000 | Chronological 80/20 |

### Model Selection Rationale:
* **Selected Model:** **Random Forest Regressor** (`n_estimators=300`, `min_samples_leaf=2`, `random_state=42`, `n_jobs=-1`).
* **Selection Metric:** Selected on the basis of **lowest Root Mean Squared Error (RMSE = 8.2731 km)** and highest coefficient of determination ($R^2 = 0.98804$).
* **Why Random Forest Outperformed Linear Regression on RMSE:**
  While Linear Regression achieved a slightly lower Mean Absolute Error ($6.116\text{ km}$ vs $6.306\text{ km}$), the ground-truth relationship contains a non-linear threshold (the flat floor at $5.00\text{ km}$ affecting 312 records). Linear regression tends to underpredict into negative values at this boundary, producing larger tail residuals that penalize RMSE. The Random Forest ensemble effectively adapts to the flat threshold cutoff.
* **KNN Regression:** High error ($\text{RMSE} = 33.294\text{ km}$) due to curse of dimensionality over 10 uninformative noise features and distance metric sensitivity.

---

## 4. Feature Importance & Technical Interpretation

Using impurity-based feature importances extracted from the fitted Random Forest pipeline:

1. `battery_soc`: **94.45%** (primary linear driver in the generator formula)
2. `battery_health`: **2.96%** (secondary positive driver)
3. `power_consumption`: **1.04%** (negative consumption penalty)
4. `road_gradient`: **0.67%** (terrain elevation resistance)
5. `speed`: **0.32%** (aerodynamic/velocity resistance)
6. Remaining 10 features (`voltage`, `current`, `vehicle_load`, `temperature`, `humidity`, etc.): **$< 0.08\%$ each**.

The model successfully recovered the true mathematical dependency hierarchy of the underlying generator while attenuating irrelevant inputs.

---

## 5. Artifact & Deliverable Locations

* **Trained Pipeline:** [`src/best_model_pipeline.joblib`](file:///d:/RAVI/PROJECTS/Software's/EV%20Prediction/DashBoard/src/best_model_pipeline.joblib)
* **Metrics Table:** [`src/model_metrics.csv`](file:///d:/RAVI/PROJECTS/Software's/EV%20Prediction/DashBoard/src/model_metrics.csv)
* **Holdout Predictions:** [`src/best_model_predictions.csv`](file:///d:/RAVI/PROJECTS/Software's/EV%20Prediction/DashBoard/src/best_model_predictions.csv)
* **Feature Importances:** [`src/random_forest_feature_importance.csv`](file:///d:/RAVI/PROJECTS/Software's/EV%20Prediction/DashBoard/src/random_forest_feature_importance.csv)
* **Interactive Dashboard:** [`dashboard/app.py`](file:///d:/RAVI/PROJECTS/Software's/EV%20Prediction/DashBoard/dashboard/app.py)
