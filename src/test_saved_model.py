
from pathlib import Path
import pandas as pd
import joblib

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "src" / "best_model_pipeline.joblib"
DATA_PATH = PROJECT_DIR / "data" / "EV_Range_Prediction_Dataset_2000.csv"

print("=" * 55)
print("SAVED MODEL VERIFICATION")
print("=" * 55)

# Check files
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Load trained pipeline
model = joblib.load(MODEL_PATH)

print("\n1. MODEL LOADED SUCCESSFULLY")
print("Model type:", type(model))

# Load dataset
df = pd.read_csv(DATA_PATH)

# Prepare sample input
X = df.drop(columns=["remaining_range", "timestamp"])
y = df["remaining_range"]

# Take five records for testing
sample = X.tail(5)
actual = y.tail(5)

# Generate predictions
predictions = model.predict(sample)

# Compare predictions
results = pd.DataFrame({
    "Actual Range (km)": actual.to_numpy(),
    "Predicted Range (km)": predictions
})

print("\n2. SAMPLE PREDICTIONS")
print(results.round(2).to_string(index=False))

print("\n3. VERIFICATION")
print("Prediction count:", len(predictions))
print("All predictions finite:", pd.Series(predictions).map(
    lambda x: pd.notna(x) and abs(x) != float("inf")
).all())

print("\nMODEL VERIFICATION COMPLETED")
print("=" * 55)
