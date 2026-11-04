
from pathlib import Path
import pandas as pd

# Locate the main project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset location
DATASET_PATH = PROJECT_DIR / "data" / "EV_Range_Prediction_Dataset_2000.csv"

print("=" * 60)
print("EV BATTERY PROJECT - DATASET VERIFICATION")
print("=" * 60)

# Check dataset existence
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}\n"
        "Check the filename and data folder."
    )

# Load dataset
df = pd.read_csv(DATASET_PATH)

# 1. Dataset dimensions
print("\n1. DATASET DIMENSIONS")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Column names
print("\n2. COLUMN NAMES")
for column in df.columns:
    print("-", column)

# 3. Data types
print("\n3. DATA TYPES")
print(df.dtypes)

# 4. Missing values
print("\n4. MISSING VALUES")
print(df.isnull().sum())

# 5. Duplicate rows
print("\n5. DUPLICATE ROWS")
print(df.duplicated().sum())

# 6. Target variable
target = "remaining_range"

print("\n6. TARGET VARIABLE")
if target in df.columns:
    print(df[target].describe())
else:
    print(f"ERROR: {target} column is missing.")

# 7. Timestamp verification
print("\n7. TIMESTAMP VERIFICATION")
if "timestamp" in df.columns:
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    print("Invalid timestamps:", timestamps.isna().sum())
    print("Earliest timestamp:", timestamps.min())
    print("Latest timestamp:", timestamps.max())
    print("Already chronological:", timestamps.is_monotonic_increasing)

print("\n" + "=" * 60)
print("DATASET VERIFICATION COMPLETED")
print("=" * 60)
