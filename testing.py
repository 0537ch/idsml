import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
data = pd.read_csv("B:/idspy/archive/02-14-2018.csv")

# Convert 'Timestamp' to datetime
data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")

# Drop the 'Timestamp' column and rows with NaN in the label
data = data.drop(columns=["Timestamp"]).dropna(subset=["Label"])

# Separate features and target
X = data.drop(columns=["Label"])
y = data["Label"]

# More thorough data cleaning
def clean_infinite_values(df):
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For each column, replace NaN with median instead of mean
    # (median is more robust to outliers)
    for column in df.columns:
        df[column] = df[column].fillna(df[column].median())
        
    # Ensure all values are within a reasonable range
    # Replace extreme values with column median
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        df[column] = df[column].clip(lower_bound, upper_bound)
    
    return df

# Clean the data
X = clean_infinite_values(X)

# Verify no infinite values remain
assert not np.any(np.isinf(X.values)), "Infinite values found in data"
assert not np.any(np.isnan(X.values)), "NaN values found in data"

# Convert data to float32 explicitly
X = X.astype(np.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nModel Performance:")
print(classification_report(y_test, y_pred))

# Save the model
with open("ids_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully as 'ids_model.pkl'")