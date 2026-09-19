import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Styling Configuration for Google Colab
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'figure.titlesize': 14,
    'axes.labelsize': 12,
    'axes.titlesize': 12
})

print("="*80)
print("PHASE 1: DATASET INSPECTION & MISSING VALUE IMPUTATION")
print("="*80)

# Load Primary ML Dataset
df = pd.read_csv('EV_Range_Prediction_Dataset_2000.csv')

print("\n--- Primary ML Dataset Info (.info) ---")
df.info()

print("\n--- First 5 Rows (.head) ---")
print(df.head())

print("\n--- Descriptive Statistics (.describe) ---")
print(df.describe().T.round(2))

# Synthetic Dataset Missing Value Imputation Demonstration
try:
    df_synth = pd.read_csv('ev_battery_synth-1.csv')
    num_synth = df_synth.select_dtypes(include=[np.number]).columns

    print("\n--- Missing Value Handling Strategy (Mean vs Median) ---")
    for col in num_synth:
        if df_synth[col].isnull().sum() > 0:
            skew_val = df_synth[col].skew()
            if abs(skew_val) > 0.5:
                fill_val = df_synth[col].median()
                strategy = "Median (High Skewness)"
            else:
                fill_val = df_synth[col].mean()
                strategy = "Mean (Symmetric)"
            df_synth[col] = df_synth[col].fillna(fill_val)
            print(f"Column '{col:25}': Imputed using {strategy:22} | Value = {fill_val:.2f}")
except Exception as e:
    print("\nSynthetic dataset file not found or skipped:", e)


# ==============================================================================
# PHASE 2: COMPLETE EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n" + "="*80)
print("PHASE 2: GENERATING COMPLETE EDA VISUALIZATIONS")
print("="*80)


# ------------------------------------------------------------------------------
# 1. UNIVARIATE ANALYSIS (Individual Feature Distributions)
# ------------------------------------------------------------------------------
print("1. Generating Univariate Plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Target Distribution
sns.histplot(df['remaining_range'], kde=True, ax=axes[0, 0], color='#2b5c8f', bins=30)
axes[0, 0].set_title('Target Distribution: Remaining Range (km)', fontweight='bold')
axes[0, 0].set_xlabel('Remaining Range (km)')

# Battery SOC Distribution
sns.histplot(df['battery_soc'], kde=True, ax=axes[0, 1], color='#27ae60', bins=30)
axes[0, 1].set_title('Feature Distribution: Battery SOC (%)', fontweight='bold')
axes[0, 1].set_xlabel('Battery SOC (%)')

# Categorical: Driving Style
sns.countplot(data=df, x='driving_style', ax=axes[1, 0], palette='Blues_d')
axes[1, 0].set_title('Categorical Analysis: Driving Style', fontweight='bold')
axes[1, 0].set_xlabel('Driving Style')

# Categorical: Traffic Density
sns.countplot(data=df, x='traffic_density', ax=axes[1, 1], palette='Greens_d')
axes[1, 1].set_title('Categorical Analysis: Traffic Density', fontweight='bold')
axes[1, 1].set_xlabel('Traffic Density')

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 2. BIVARIATE ANALYSIS (Basic Relationships)
# ------------------------------------------------------------------------------
print("2. Generating Bivariate Plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Range vs Battery SOC by Driving Style
sns.scatterplot(data=df, x='battery_soc', y='remaining_range', hue='driving_style', alpha=0.7, ax=axes[0], palette='Set2')
axes[0].set_title('Remaining Range vs. Battery SOC (by Driving Style)', fontweight='bold')
axes[0].set_xlabel('Battery SOC (%)')
axes[0].set_ylabel('Remaining Range (km)')

# Boxplot Across Driving Styles & Traffic Density
sns.boxplot(data=df, x='driving_style', y='remaining_range', hue='traffic_density', ax=axes[1], palette='Set3')
axes[1].set_title('Remaining Range Across Driving Styles & Traffic Density', fontweight='bold')
axes[1].set_xlabel('Driving Style')
axes[1].set_ylabel('Remaining Range (km)')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 3. ADVANCED MULTI-FACTOR ANALYSIS PLOTS
# ------------------------------------------------------------------------------
print("3. Generating Advanced Multi-Factor Analysis Grid...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Speed vs Power Consumption
sns.scatterplot(
    data=df, x='speed', y='power_consumption',
    hue='driving_style', alpha=0.7, ax=axes[0, 0], palette='deep'
)
axes[0, 0].set_title('Vehicle Speed vs. Power Consumption (by Driving Style)', fontweight='bold')
axes[0, 0].set_xlabel('Speed (km/h)')
axes[0, 0].set_ylabel('Power Consumption (kW)')

# Temp vs Range with Trendline
sns.regplot(
    data=df, x='temperature', y='remaining_range', ax=axes[0, 1],
    scatter_kws={'alpha': 0.3, 'color': '#e74c3c'}, line_kws={'color': 'darkred'}
)
axes[0, 1].set_title('Temperature (°C) vs. Remaining Range (with Trendline)', fontweight='bold')
axes[0, 1].set_xlabel('Ambient Temperature (°C)')
axes[0, 1].set_ylabel('Remaining Range (km)')

# Categorized SOC Bins vs Remaining Range across Traffic Density
df['soc_bin'] = pd.cut(df['battery_soc'], bins=[0, 30, 70, 100], labels=['Low (<30%)', 'Medium (30-70%)', 'High (>70%)'])
sns.boxplot(
    data=df, x='soc_bin', y='remaining_range',
    hue='traffic_density', ax=axes[1, 0], palette='Set2'
)
axes[1, 0].set_title('Remaining Range across SOC Bins & Traffic Density', fontweight='bold')
axes[1, 0].set_xlabel('Battery SOC Category')
axes[1, 0].set_ylabel('Remaining Range (km)')

# Electrical Parameter Violin Plot
sns.violinplot(
    data=df[['power_consumption', 'voltage', 'current']],
    ax=axes[1, 1], palette='Pastel1'
)
axes[1, 1].set_title('Spread of Electrical Parameters (Power, Voltage, Current)', fontweight='bold')

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 4. MULTIVARIATE ANALYSIS (Correlation Heatmap)
# ------------------------------------------------------------------------------
print("4. Generating Multivariate Correlation Heatmap...")
plt.figure(figsize=(12, 9))
num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()

# Mask upper triangle for clean visualization
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Multivariate Correlation Heatmap (EV Operational Parameters)', fontweight='bold')
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 5. FEATURE PAIRPLOT MATRIX
# ------------------------------------------------------------------------------
print("5. Generating Feature Pairplot Matrix...")
sns_pair = sns.pairplot(
    df[['remaining_range', 'battery_soc', 'power_consumption', 'speed', 'battery_health']],
    corner=True, diag_kind='kde', plot_kws={'alpha': 0.4, 's': 20, 'color': '#2980b9'}
)
sns_pair.fig.suptitle('Pairwise Feature Interactions Matrix', y=1.02, fontweight='bold')
plt.show()
