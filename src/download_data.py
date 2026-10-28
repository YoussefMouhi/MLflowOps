from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# ==========================================
# DOWNLOAD AI4I PREDICTIVE MAINTENANCE DATA
# ==========================================

print("Downloading AI4I 2020 Predictive Maintenance Dataset...")

dataset = fetch_ucirepo(id=601)

X = dataset.data.features.copy()
y = dataset.data.targets.copy()

# Combine features and target columns
df = pd.concat([X, y], axis=1)

# Create output directory
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

# Save dataset locally
output_path = output_dir / "ai4i2020.csv"

df.to_csv(output_path, index=False)

print("\nDataset downloaded successfully!")
print(f"Saved to: {output_path}")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())