
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "EV_Range_Prediction_Dataset_2000.csv"
MODEL_PATH = PROJECT_DIR / "src" / "best_model_pipeline.joblib"
METRICS_PATH = PROJECT_DIR / "src" / "model_metrics.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_DIR / "src" / "random_forest_feature_importance.csv"
PREDICTIONS_PATH = PROJECT_DIR / "src" / "best_model_predictions.csv"

NUMERIC_FEATURES = [
    "battery_soc",
    "voltage",
    "current",
    "power_consumption",
    "speed",
    "distance_traveled",
    "road_gradient",
    "temperature",
    "humidity",
    "wind_speed",
    "vehicle_load",
    "battery_health",
    "regen_braking",
]

CATEGORICAL_FEATURES = [
    "traffic_density",
    "driving_style",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "remaining_range"

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EV Battery Performance Dashboard",
    page_icon="🔋",
    layout="wide",
)

# --------------------------------------------------
# CACHED DATA & MODEL LOADERS
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_metrics():
    if not METRICS_PATH.exists():
        return None, "File not found"
    try:
        data = pd.read_csv(METRICS_PATH)
        required = {"model", "MAE_km", "RMSE_km", "R2"}
        if not required.issubset(data.columns):
            missing = required - set(data.columns)
            return None, f"Missing required columns: {sorted(missing)}"
        return data, None
    except Exception as exc:
        return None, f"Error reading file: {exc}"


@st.cache_data
def load_feature_importance():
    if not FEATURE_IMPORTANCE_PATH.exists():
        return None, "File not found"
    try:
        data = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        required = {"feature", "importance"}
        if not required.issubset(data.columns):
            missing = required - set(data.columns)
            return None, f"Missing required columns: {sorted(missing)}"
        return data, None
    except Exception as exc:
        return None, f"Error reading file: {exc}"


@st.cache_data
def load_best_model_predictions():
    if not PREDICTIONS_PATH.exists():
        return None, "File not found"
    try:
        data = pd.read_csv(PREDICTIONS_PATH)
        required = {"actual_km", "predicted_km"}
        if not required.issubset(data.columns):
            missing = required - set(data.columns)
            return None, f"Missing required columns: {sorted(missing)}"

        data["actual_km"] = pd.to_numeric(data["actual_km"], errors="coerce")
        data["predicted_km"] = pd.to_numeric(data["predicted_km"], errors="coerce")
        clean_data = data.dropna(subset=["actual_km", "predicted_km"]).copy()

        if len(clean_data) == 0:
            return None, "No valid numeric prediction rows found"

        return clean_data, None
    except Exception as exc:
        return None, f"Error reading file: {exc}"


# --------------------------------------------------
# PRE-FLIGHT VERIFICATION
# --------------------------------------------------

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()

try:
    df = load_dataset()
    model = load_model()
except Exception as error:
    st.error(f"Could not load project files: {error}")
    st.stop()

missing_columns = [
    column for column in FEATURES + [TARGET]
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing required dataset columns: {missing_columns}")
    st.stop()

# --------------------------------------------------
# APP HEADER & STYLING
# --------------------------------------------------

st.title("EV Battery Performance & Range Prediction")
st.caption(
    "Machine Learning Dashboard | Data Science and Visualization | 2026–2027"
)

# Custom minimal CSS for clean navbar presentation
st.markdown(
    """
    <style>
    div[data-testid="stSegmentedControl"] {
        display: flex;
        justify-content: center;
        margin-top: 0.25rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        margin-top: 0.25rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

NAV_PAGES = [
    "Home / Overview",
    "Range Predictor",
    "Battery Analytics",
    "Model Insights",
]

if hasattr(st, "segmented_control"):
    selected_page = st.segmented_control(
        "Navigation",
        options=NAV_PAGES,
        default="Home / Overview",
        required=True,
        label_visibility="collapsed",
    )
else:
    selected_page = st.radio(
        "Navigation",
        options=NAV_PAGES,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

if not selected_page:
    selected_page = "Home / Overview"

st.divider()

# ==================================================
# PAGE 1: HOME / OVERVIEW
# ==================================================

if selected_page == "Home / Overview":

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Average Battery SOC", f"{df['battery_soc'].mean():.1f}%")
    col3.metric("Average Battery Health", f"{df['battery_health'].mean():.1f}%")
    col4.metric("Average Remaining Range", f"{df[TARGET].mean():.1f} km")

    st.markdown("### Application Purpose")
    st.write(
        "Accurate remaining driving range prediction is a core requirement for modern "
        "electric vehicles (EVs) to mitigate range anxiety, optimize trip routing, and maximize battery lifespan. "
        "This interactive dashboard leverages machine learning trained on multi-dimensional telemetry—incorporating "
        "battery states, electrical draw, powertrain dynamics, terrain gradient, environmental weather factors, and "
        "driving behavior—to deliver real-time range estimates and deep performance diagnostics."
    )

    st.markdown("### Key Findings & Research Summary")
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown(
            """
            - **Primary Range Driver:** Battery State of Charge (SOC) contributes over **94%** of feature importance, exhibiting a strong monotonic linear relationship with remaining distance.
            - **Secondary Influences:** Battery health (~2.96%) and instantaneous power consumption (~1.04%) exert notable secondary impacts on effective range.
            - **Dynamic Operating Factors:** Road gradient, vehicle velocity, and auxiliary loads (temperature/humidity/speed) modulate consumption during active driving cycles.
            """
        )

    with info_col2:
        st.markdown(
            """
            - **Selected Architecture:** A tuned **Random Forest Regressor** (300 estimators, leaf size 2) achieved superior generalisation on chronological test data.
            - **Validation Performance:** Holds the lowest test error (**RMSE = 8.2731 km**, **MAE = 6.3057 km**) with an **$R^2$ of 0.98804**, outperforming linear and neighbor baselines.
            - **Chronological Holdout:** Evaluation adheres to a temporal 80:20 split (first 1,600 observations for training, final 400 for testing) to prevent temporal data leakage.
            """
        )

    st.markdown("### Dashboard Navigation Guide")
    nav_c1, nav_c2, nav_c3 = st.columns(3)
    with nav_c1:
        st.markdown(
            "**⚡ Range Predictor**\n\n"
            "Configure real-time battery, vehicular, and environmental inputs to generate instantaneous range estimates via the trained Random Forest pipeline."
        )
    with nav_c2:
        st.markdown(
            "**📊 Battery Analytics**\n\n"
            "Explore target range distributions, SOC correlation scatterplots, and thorough prediction diagnostic curves (Actual vs. Predicted, Residuals, and Error histograms)."
        )
    with nav_c3:
        st.markdown(
            "**🧠 Model Insights**\n\n"
            "Review benchmark comparison tables across all candidate regressors, comparative RMSE charts, impurity-based feature importances, and pipeline specifications."
        )

# ==================================================
# PAGE 2: RANGE PREDICTOR
# ==================================================

elif selected_page == "Range Predictor":

    st.header("Predict Remaining Driving Range")

    st.write(
        "Enter the battery and vehicle conditions below to estimate "
        "the remaining driving range."
    )

    st.info(
        "The prediction uses the trained Random Forest pipeline, "
        "including its fitted preprocessing steps."
    )

    with st.form("prediction_form"):

        st.subheader("Battery Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            battery_soc = st.number_input(
                "Battery SOC (%)",
                min_value=float(df["battery_soc"].min()),
                max_value=float(df["battery_soc"].max()),
                value=float(df["battery_soc"].median()),
                step=1.0,
            )

        with c2:
            battery_health = st.number_input(
                "Battery Health (%)",
                min_value=float(df["battery_health"].min()),
                max_value=float(df["battery_health"].max()),
                value=float(df["battery_health"].median()),
                step=1.0,
            )

        with c3:
            voltage = st.number_input(
                "Voltage (V)",
                min_value=float(df["voltage"].min()),
                max_value=float(df["voltage"].max()),
                value=float(df["voltage"].median()),
                step=1.0,
            )

        st.subheader("Electrical and Driving Conditions")

        c4, c5, c6 = st.columns(3)

        with c4:
            current = st.number_input(
                "Current (A)",
                min_value=float(df["current"].min()),
                max_value=float(df["current"].max()),
                value=float(df["current"].median()),
                step=1.0,
            )

        with c5:
            power_consumption = st.number_input(
                "Power Consumption",
                min_value=float(df["power_consumption"].min()),
                max_value=float(df["power_consumption"].max()),
                value=float(df["power_consumption"].median()),
                step=0.1,
            )

        with c6:
            speed = st.number_input(
                "Vehicle Speed (km/h)",
                min_value=float(df["speed"].min()),
                max_value=float(df["speed"].max()),
                value=float(df["speed"].median()),
                step=1.0,
            )

        c7, c8, c9 = st.columns(3)

        with c7:
            distance_traveled = st.number_input(
                "Distance Travelled",
                min_value=float(df["distance_traveled"].min()),
                max_value=float(df["distance_traveled"].max()),
                value=float(df["distance_traveled"].median()),
                step=0.1,
            )

        with c8:
            road_gradient = st.number_input(
                "Road Gradient",
                min_value=float(df["road_gradient"].min()),
                max_value=float(df["road_gradient"].max()),
                value=float(df["road_gradient"].median()),
                step=0.1,
            )

        with c9:
            vehicle_load = st.number_input(
                "Vehicle Load",
                min_value=float(df["vehicle_load"].min()),
                max_value=float(df["vehicle_load"].max()),
                value=float(df["vehicle_load"].median()),
                step=1.0,
            )

        st.subheader("Environmental Conditions")

        c10, c11, c12 = st.columns(3)

        with c10:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=float(df["temperature"].min()),
                max_value=float(df["temperature"].max()),
                value=float(df["temperature"].median()),
                step=0.5,
            )

        with c11:
            humidity = st.number_input(
                "Humidity (%)",
                min_value=float(df["humidity"].min()),
                max_value=float(df["humidity"].max()),
                value=float(df["humidity"].median()),
                step=1.0,
            )

        with c12:
            wind_speed = st.number_input(
                "Wind Speed",
                min_value=float(df["wind_speed"].min()),
                max_value=float(df["wind_speed"].max()),
                value=float(df["wind_speed"].median()),
                step=0.1,
            )

        st.subheader("Driving Behaviour")

        c13, c14 = st.columns(2)

        with c13:
            traffic_density = st.selectbox(
                "Traffic Density",
                options=sorted(df["traffic_density"].dropna().unique().tolist()),
            )

        with c14:
            driving_style = st.selectbox(
                "Driving Style",
                options=sorted(df["driving_style"].dropna().unique().tolist()),
            )

        regen_braking = st.slider(
            "Regenerative Braking",
            min_value=float(df["regen_braking"].min()),
            max_value=float(df["regen_braking"].max()),
            value=float(df["regen_braking"].median()),
            step=1.0,
        )

        submitted = st.form_submit_button(
            "Predict Remaining Range",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------
    # GENERATE PREDICTION
    # --------------------------------------------------

    if submitted:

        input_data = pd.DataFrame([{
            "battery_soc": battery_soc,
            "voltage": voltage,
            "current": current,
            "power_consumption": power_consumption,
            "speed": speed,
            "distance_traveled": distance_traveled,
            "road_gradient": road_gradient,
            "traffic_density": traffic_density,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "driving_style": driving_style,
            "vehicle_load": vehicle_load,
            "battery_health": battery_health,
            "regen_braking": regen_braking,
        }], columns=FEATURES)

        try:
            prediction = float(model.predict(input_data)[0])

            if not np.isfinite(prediction):
                st.error("The model returned an invalid prediction.")
            else:
                st.success("Prediction generated successfully!")

                st.subheader("Estimated Remaining Driving Range")

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.metric(
                        "Predicted Range",
                        f"{prediction:.2f} km",
                    )

                with result_col2:
                    st.metric(
                        "Battery SOC",
                        f"{battery_soc:.1f}%",
                    )

                st.caption(
                    "This is a model estimate based on the supplied inputs, "
                    "not a guaranteed real-world vehicle range."
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# ==================================================
# PAGE 3: BATTERY ANALYTICS
# ==================================================

elif selected_page == "Battery Analytics":

    st.header("Battery Performance Analytics")

    chart1, chart2 = st.columns(2)

    with chart1:
        fig1 = px.histogram(
            df,
            x=TARGET,
            nbins=30,
            title="Remaining Range Distribution",
            labels={TARGET: "Remaining Range (km)"},
        )
        st.plotly_chart(fig1, use_container_width=True)

    with chart2:
        fig2 = px.scatter(
            df,
            x="battery_soc",
            y=TARGET,
            color="driving_style",
            title="Battery SOC vs Remaining Range",
            labels={
                "battery_soc": "Battery SOC (%)",
                TARGET: "Remaining Range (km)",
            },
            opacity=0.65,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # PREDICTION DIAGNOSTICS
    # --------------------------------------------------

    st.header("Prediction Diagnostics")

    pred_df, pred_error = load_best_model_predictions()

    if pred_error:
        st.warning(
            f"⚠️ Prediction diagnostics could not be loaded ({pred_error}). "
            f"Ensure `{PREDICTIONS_PATH.name}` exists in `src/`."
        )
    else:
        actual = pred_df["actual_km"]
        predicted = pred_df["predicted_km"]
        residuals = actual - predicted
        abs_errors = np.abs(residuals)

        pred_df["residual_km"] = residuals
        pred_df["abs_error_km"] = abs_errors

        n_test = len(pred_df)
        diag_mae = float(abs_errors.mean())
        diag_rmse = float(np.sqrt((residuals ** 2).mean()))
        diag_max_error = float(abs_errors.max())

        st.subheader("Test Set Error Summary")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        d_col1.metric("Test Predictions", f"{n_test:,}")
        d_col2.metric("Mean Absolute Error (MAE)", f"{diag_mae:.3f} km")
        d_col3.metric("Root Mean Squared Error (RMSE)", f"{diag_rmse:.3f} km")
        d_col4.metric("Maximum Absolute Error", f"{diag_max_error:.3f} km")

        d_chart1, d_chart2 = st.columns(2)

        # 1. ACTUAL VS PREDICTED
        with d_chart1:
            min_val = float(min(actual.min(), predicted.min()))
            max_val = float(max(actual.max(), predicted.max()))

            fig_avp = px.scatter(
                pred_df,
                x="actual_km",
                y="predicted_km",
                title="Actual vs. Predicted Remaining Range",
                labels={
                    "actual_km": "Actual Range (km)",
                    "predicted_km": "Predicted Range (km)",
                },
                opacity=0.7,
                color_discrete_sequence=["#00E676"],
            )
            fig_avp.add_shape(
                type="line",
                x0=min_val,
                y0=min_val,
                x1=max_val,
                y1=max_val,
                line=dict(color="#E0E0E0", dash="dash", width=2),
            )
            fig_avp.update_layout(
                template="plotly_dark",
                margin=dict(t=50, b=40, l=40, r=40),
            )
            st.plotly_chart(fig_avp, use_container_width=True)

        # 2. RESIDUAL ANALYSIS
        with d_chart2:
            fig_res = px.scatter(
                pred_df,
                x="predicted_km",
                y="residual_km",
                title="Residual Analysis (Actual - Predicted)",
                labels={
                    "predicted_km": "Predicted Range (km)",
                    "residual_km": "Residual (km)",
                },
                opacity=0.7,
                color_discrete_sequence=["#2589A6"],
            )
            fig_res.add_hline(
                y=0,
                line_dash="dash",
                line_color="#E0E0E0",
                annotation_text="Zero Error (y=0)",
                annotation_position="bottom right",
            )
            fig_res.update_layout(
                template="plotly_dark",
                margin=dict(t=50, b=40, l=40, r=40),
            )
            st.plotly_chart(fig_res, use_container_width=True)

        # 3. PREDICTION ERROR DISTRIBUTION
        fig_hist = px.histogram(
            pred_df,
            x="abs_error_km",
            nbins=30,
            title="Prediction Error Distribution (|Actual - Predicted|)",
            labels={"abs_error_km": "Absolute Prediction Error (km)"},
            color_discrete_sequence=["#D56B35"],
        )
        fig_hist.update_layout(
            template="plotly_dark",
            xaxis_title="Absolute Error (km)",
            yaxis_title="Observation Count",
            margin=dict(t=50, b=40, l=40, r=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # DIAGNOSTIC EXPLANATION
        st.info(
            "**Diagnostic Interpretation:**\n\n"
            "- **Residuals close to zero** indicate smaller prediction errors and higher model accuracy.\n"
            "- **Patterns in residuals** (such as curved trends or uneven dispersion) can indicate systematic prediction errors or bias.\n"
            "- **Scope:** These results describe model performance on the held-out test dataset and do not prove real-world vehicle accuracy under unobserved road or weather conditions."
        )

# ==================================================
# PAGE 4: MODEL INSIGHTS
# ==================================================

elif selected_page == "Model Insights":

    st.header("Model Performance & Comparison")

    metrics_df, metrics_error = load_model_metrics()

    if metrics_error:
        st.warning(
            f"⚠️ Model metrics could not be loaded ({metrics_error}). "
            f"Ensure `{METRICS_PATH.name}` exists in `src/`."
        )
    else:
        metrics_sorted = metrics_df.sort_values("RMSE_km", ascending=True).copy()
        best_row = metrics_sorted.iloc[0]
        best_model_name = best_row["model"]

        comp_col1, comp_col2 = st.columns([1, 1])

        with comp_col1:
            st.subheader("Model Evaluation Table")

            display_table = metrics_sorted[["model", "MAE_km", "RMSE_km", "R2"]].copy()
            display_table["MAE (km)"] = display_table["MAE_km"].apply(lambda v: f"{v:.3f}")
            display_table["RMSE (km)"] = display_table["RMSE_km"].apply(lambda v: f"{v:.3f}")
            display_table["R² Score"] = display_table["R2"].apply(lambda v: f"{v:.4f}")
            display_table["Status"] = display_table["model"].apply(
                lambda m: "⭐ Best (Lowest RMSE)" if m == best_model_name else "Candidate"
            )
            display_table = display_table[["model", "MAE (km)", "RMSE (km)", "R² Score", "Status"]].rename(
                columns={"model": "Model Name"}
            )

            st.dataframe(display_table, use_container_width=True, hide_index=True)

            st.success(
                f"**Best Performing Model:** **{best_model_name}** achieved the lowest test "
                f"RMSE of **{best_row['RMSE_km']:.3f} km**, MAE of **{best_row['MAE_km']:.3f} km**, "
                f"and R² of **{best_row['R2']:.4f}**."
            )

        with comp_col2:
            st.subheader("RMSE Comparison Across Models")

            metrics_sorted["Status"] = metrics_sorted["model"].apply(
                lambda m: "⭐ Best Model (Lowest RMSE)" if m == best_model_name else "Candidate Model"
            )

            fig_comp = px.bar(
                metrics_sorted,
                x="model",
                y="RMSE_km",
                color="Status",
                color_discrete_map={
                    "⭐ Best Model (Lowest RMSE)": "#00E676",
                    "Candidate Model": "#2589A6",
                },
                text=metrics_sorted["RMSE_km"].apply(lambda v: f"{v:.3f} km"),
                title="Model Comparison by RMSE (Holdout Test Set)",
                labels={
                    "model": "Regression Model",
                    "RMSE_km": "RMSE (km)",
                    "Status": "Evaluation Status",
                },
            )
            fig_comp.update_traces(textposition="outside")
            fig_comp.update_layout(
                template="plotly_dark",
                xaxis_title="Regression Model",
                yaxis_title="Root Mean Squared Error (km)",
                legend_title_text="",
                margin=dict(t=50, b=40, l=40, r=40),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # RANDOM FOREST FEATURE IMPORTANCE
    # --------------------------------------------------

    st.header("Random Forest Feature Importance")

    fi_df, fi_error = load_feature_importance()

    if fi_error:
        st.warning(
            f"⚠️ Feature importance could not be loaded ({fi_error}). "
            f"Ensure `{FEATURE_IMPORTANCE_PATH.name}` exists in `src/`."
        )
    else:
        top10_fi = fi_df.sort_values("importance", ascending=False).head(10).copy()
        top10_fi["feature_clean"] = (
            top10_fi["feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
            .str.replace("_", " ", regex=False)
            .str.title()
        )
        top10_fi["importance_pct"] = top10_fi["importance"] * 100

        fig_fi = px.bar(
            top10_fi,
            x="importance",
            y="feature_clean",
            orientation="h",
            text=top10_fi["importance_pct"].apply(
                lambda v: f"{v:.2f}%" if v >= 0.1 else f"{v:.3f}%"
            ),
            title="Top 10 Feature Importances (Random Forest Impurity Reduction)",
            labels={
                "importance": "Relative Importance",
                "feature_clean": "Feature",
            },
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(
            template="plotly_dark",
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_showscale=False,
            xaxis_title="Relative Feature Importance",
            yaxis_title="",
            margin=dict(t=50, b=40, l=40, r=40),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # DYNAMIC MODEL ARCHITECTURE & EVALUATION DETAILS
    # --------------------------------------------------

    st.header("Model Information")

    if metrics_df is not None and metrics_error is None:
        best_metric_row = metrics_df.sort_values("RMSE_km").iloc[0]
        selected_model_name = best_metric_row["model"]
        best_mae = best_metric_row["MAE_km"]
        best_rmse = best_metric_row["RMSE_km"]
        best_r2 = best_metric_row["R2"]
        split_desc = best_metric_row.get("split", "Chronological 80:20")
        train_count = (
            int(best_metric_row["train_n"])
            if "train_n" in best_metric_row and pd.notna(best_metric_row["train_n"])
            else 1600
        )
        test_count = (
            int(best_metric_row["test_n"])
            if "test_n" in best_metric_row and pd.notna(best_metric_row["test_n"])
            else 400
        )

        st.write(f"**Selected model (Lowest RMSE):** {selected_model_name}")
        st.write(f"**Model pipeline file:** `{MODEL_PATH.relative_to(PROJECT_DIR)}`")
        st.write(
            f"**Training/testing split:** "
            f"{str(split_desc).title() if isinstance(split_desc, str) else split_desc}"
        )
        st.write(f"**Training records:** {train_count:,}")
        st.write(f"**Testing records:** {test_count:,}")

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Mean Absolute Error (MAE)", f"{best_mae:.3f} km")
        m_col2.metric("Root Mean Squared Error (RMSE)", f"{best_rmse:.3f} km")
        m_col3.metric("R² Score", f"{best_r2:.4f}")
    else:
        st.write("**Selected model:** Random Forest Regressor")
        st.write(f"**Model pipeline file:** `{MODEL_PATH.relative_to(PROJECT_DIR)}`")
        st.write("**Training/testing split:** Chronological 80:20")
        st.write("**Training records:** 1,600")
        st.write("**Testing records:** 400")
        if metrics_error:
            st.warning(f"⚠️ Dynamic metrics unavailable: {metrics_error}")

# ==================================================
# FOOTER
# ==================================================

st.divider()
st.caption(
    "Project: Electric Vehicle Battery Performance and Range Prediction | "
    "Department of ECE | Data Science and Visualization"
)
