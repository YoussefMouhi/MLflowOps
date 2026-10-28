from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ==================================================
# CONFIGURATION
# ==================================================

RAW_DATA_PATH = Path("data/raw/ai4i2020.csv")
OUTPUT_DIR = Path("data/processed")

FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

TARGET = "Machine failure"

RANDOM_STATE = 42


# ==================================================
# LOAD DATA
# ==================================================

print("\nLoading AI4I predictive maintenance dataset...")

df = pd.read_csv(RAW_DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ==================================================
# SELECT FEATURES AND TARGET
# ==================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("\nSelected features:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
print((y.value_counts(normalize=True) * 100).round(2))


# ==================================================
# TRAIN / VALIDATION / TEST SPLIT
# ==================================================

# 70% training, 15% validation, 15% testing

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE,
)


# ==================================================
# SAVE DATASETS
# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_split(X_split, y_split, filename):
    split_df = X_split.copy()
    split_df[TARGET] = y_split
    split_df.to_csv(OUTPUT_DIR / filename, index=False)


save_split(X_train, y_train, "train.csv")
save_split(X_val, y_val, "validation.csv")
save_split(X_test, y_test, "test.csv")


# ==================================================
# SUMMARY
# ==================================================

print("\n" + "=" * 50)
print("DATA PREPROCESSING COMPLETED")
print("=" * 50)

print(f"Training samples  : {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples   : {len(X_test)}")

print("\nFailure distribution by split:")

for name, labels in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Testing", y_test),
]:
    print(
        f"{name}: "
        f"{int(labels.sum())} failures "
        f"({labels.mean() * 100:.2f}%)"
    )

print("\nFiles saved to data/processed/")
print("\nPreprocessing completed successfully!")
