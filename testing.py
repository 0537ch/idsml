import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def create_project_folders():
    # Mengubah base directory ke B:\idspy\train_result
    base_dir = r'B:\idspy\train_result'
    
    # Membuat direktori utama jika belum ada
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join(base_dir, f'analysis_{timestamp}')
    os.makedirs(result_folder)
    
    models_folder = os.path.join(result_folder, 'models')
    plots_folder = os.path.join(result_folder, 'plots')
    reports_folder = os.path.join(result_folder, 'reports')
    
    for folder in [models_folder, plots_folder, reports_folder]:
        os.makedirs(folder)
    
    return result_folder, models_folder, plots_folder, reports_folder

def process_data(data_path):
    print("Loading and processing data...")
    data = pd.read_csv(data_path)
    
    if "Timestamp" in data.columns:
        data = data.drop(columns=["Timestamp"])
    
    X = data.drop(columns=["Label"])
    y = data["Label"]
    
    X = clean_infinite_values(X)
    X = X.astype(np.float32)
    
    return X, y

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

def train_and_save_results(data_path):
    
    result_folder, models_folder, plots_folder, reports_folder = create_project_folders()
    print(f"\nCreated folders at: {result_folder}")
    
    X, y = process_data(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    model_path = os.path.join(models_folder, 'ids_model.pkl')
    print(f"Saving model to: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    plt.figure(figsize=(10, 8))
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(plots_folder, 'confusion_matrix.png'))
    plt.close()
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(10), feature_importance['importance'][:10])
    plt.xticks(range(10), feature_importance['feature'][:10], rotation=45)
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'feature_importance.png'))
    plt.close()
    
    model_info = {
        'Number of features': X.shape[1],
        'Number of samples': X.shape[0],
        'Feature names': list(X.columns),
        'Model parameters': model.get_params()
    }
    
    with open(os.path.join(reports_folder, 'model_info.txt'), 'w') as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
    
    feature_importance.to_csv(os.path.join(reports_folder, 'feature_importance.csv'), index=False)
    
    print("\nGenerated files:")
    print(f"1. Model: {models_folder}/ids_model.pkl")
    print(f"2. Plots: {plots_folder}/confusion_matrix.png")
    print(f"         {plots_folder}/feature_importance.png")
    print(f"3. Reports: {reports_folder}/model_info.txt")
    print(f"           {reports_folder}/feature_importance.csv")
    
    return result_folder

if __name__ == "__main__":
    data_path = "B:/archive/02-14-2018.csv"
    
    print("Starting model training and result generation...")
    result_folder = train_and_save_results(data_path)
    print(f"\nAll results have been saved in: {result_folder}")
    
    print("\nList of generated files:")
    for root, dirs, files in os.walk(result_folder):
        level = root.replace(result_folder, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")