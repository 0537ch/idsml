import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os
from datetime import datetime

# Buat fungsi untuk membuat direktori jika belum ada
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

# Tentukan direktori untuk menyimpan model dan hasil
current_dir = os.path.dirname(os.path.abspath(__file__))  # Dapatkan direktori script saat ini
output_dir = os.path.join(current_dir, "model_output")    # Buat subfolder untuk output
create_directory(output_dir)

# Load dataset
print("Loading dataset...")
data = pd.read_csv("B:/archive/02-14-2018.csv")

# Convert 'Timestamp' to datetime
data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")

# Drop the 'Timestamp' column and rows with NaN in the label
data = data.drop(columns=["Timestamp"]).dropna(subset=["Label"])

# Separate features and target
X = data.drop(columns=["Label"])
y = data["Label"]

# Clean infinite values
def clean_infinite_values(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    for column in df.columns:
        df[column] = df[column].fillna(df[column].median())
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df[column] = df[column].clip(lower_bound, upper_bound)
    return df

# Clean the data
print("Cleaning data...")
X = clean_infinite_values(X)
X = X.astype(np.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Generate timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save model dengan path lengkap
model_filename = os.path.join(output_dir, f"ids_model_{timestamp}.pkl")
print(f"\nSaving model to: {model_filename}")
with open(model_filename, "wb") as f:
    pickle.dump(model, f)

# Save model info
model_info = {
    "model_path": model_filename,
    "timestamp": timestamp,
    "features": list(X.columns),
    "n_features": X.shape[1],
    "n_samples": X.shape[0]
}

info_filename = os.path.join(output_dir, f"model_info_{timestamp}.txt")
print(f"Saving model info to: {info_filename}")
with open(info_filename, "w") as f:
    for key, value in model_info.items():
        f.write(f"{key}: {value}\n")

print("\nModel dan info telah disimpan di folder:", output_dir)
print("\nDetail file yang disimpan:")
print(f"1. Model: {model_filename}")
print(f"2. Info: {info_filename}")

# Tampilkan isi folder
print("\nIsi folder output:")
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    print(f"- {file} ({file_size:.2f} MB)")