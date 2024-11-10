import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def create_project_folders():
    # Fungsi ini tetap sama seperti sebelumnya
    base_dir = r'B:\idspy\train_result'
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
    
    with open(os.path.join(result_folder, 'training_progress.txt'), 'w') as f:
        f.write("Training Progress Log\n")
    
    return result_folder, models_folder, plots_folder, reports_folder

def initialize_feature_columns(data_folder):
    """Menginisialisasi daftar kolom yang konsisten dari semua file CSV."""
    all_columns = set()
    
    # Baca header dari semua file untuk mendapatkan union dari semua kolom
    for csv_file in glob.glob(os.path.join(data_folder, "*.csv")):
        try:
            header = pd.read_csv(csv_file, nrows=0)
            all_columns.update(header.columns)
        except Exception as e:
            print(f"Warning: Could not read headers from {os.path.basename(csv_file)}: {str(e)}")
    
    # Hapus kolom yang tidak diinginkan
    columns_to_remove = {'Timestamp', 'Label'}
    feature_columns = sorted(list(all_columns - columns_to_remove))
    
    return feature_columns

def process_data_chunk(chunk, expected_features):
    """Memproses chunk data dengan daftar fitur yang diharapkan."""
    try:
        # Pastikan chunk memiliki kolom Label
        if 'Label' not in chunk.columns:
            print("Warning: Missing Label column in chunk")
            return None, None
        
        # Hapus kolom timestamp jika ada
        if 'Timestamp' in chunk.columns:
            chunk = chunk.drop(columns=['Timestamp'])
        
        # Konversi semua kolom non-numerik ke numerik
        for col in chunk.columns:
            if col != 'Label':
                try:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                except:
                    print(f"Warning: Could not convert column {col} to numeric")
        
        # Tambahkan kolom yang hilang dengan nilai NaN
        for col in expected_features:
            if col not in chunk.columns and col != 'Label':
                chunk[col] = np.nan
        
        # Pilih hanya kolom yang diharapkan plus Label
        selected_columns = expected_features + ['Label']
        chunk = chunk[selected_columns]
        
        # Pisahkan fitur dan label
        X = chunk.drop(columns=['Label'])
        y = chunk['Label']
        
        # Bersihkan nilai infinite dan handle outliers
        X = clean_infinite_values(X)
        X = X.astype(np.float32)
        
        return X, y
        
    except Exception as e:
        print(f"Error in process_data_chunk: {str(e)}")
        return None, None

def clean_infinite_values(df):
    """Membersihkan nilai infinite dan outlier dengan penanganan error yang lebih baik."""
    try:
        df = df.replace([np.inf, -np.inf], np.nan)
        
        for column in df.columns:
            # Hitung median hanya untuk nilai non-NaN
            valid_values = df[column].dropna()
            if len(valid_values) > 0:
                median_val = valid_values.median()
            else:
                median_val = 0
            
            # Isi NaN dengan median
            df[column] = df[column].fillna(median_val)
            
            # Handle outliers
            if np.issubdtype(df[column].dtype, np.number):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                df[column] = df[column].clip(lower_bound, upper_bound)
        
        return df
    except Exception as e:
        print(f"Error in clean_infinite_values: {str(e)}")
        return df

def train_incrementally(data_folder, chunk_size=50000):
    result_folder, models_folder, plots_folder, reports_folder = create_project_folders()
    print(f"\nCreated folders at: {result_folder}")
    
    # Inisialisasi daftar fitur yang konsisten
    expected_features = initialize_feature_columns(data_folder)
    print(f"Initialized with {len(expected_features)} features")
    
    model_path = os.path.join(models_folder, 'ids_model_incremental.pkl')
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    log_file = os.path.join(result_folder, 'training_progress.txt')
    total_processed = 0
    feature_importance_sum = np.zeros(len(expected_features))
    
    for csv_file in csv_files:
        print(f"\nProcessing: {os.path.basename(csv_file)}")
        
        try:
            chunks = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)
            
            for chunk_num, chunk in enumerate(chunks):
                X, y = process_data_chunk(chunk, expected_features)
                
                if X is None or y is None or X.empty or y.empty:
                    print(f"Skipping invalid chunk {chunk_num} in {os.path.basename(csv_file)}")
                    continue
                
                if X.shape[1] != len(expected_features):
                    print(f"Feature mismatch in chunk {chunk_num}: expected {len(expected_features)}, got {X.shape[1]}")
                    continue
                
                # Update model
                model.fit(X, y)
                feature_importance_sum += model.feature_importances_
                total_processed += len(X)
                
                # Log progress
                with open(log_file, 'a') as f:
                    f.write(f"\nProcessed {os.path.basename(csv_file)} - Chunk {chunk_num+1}, Records: {len(X)}")
                
                # Save model
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                print(f"Processed chunk {chunk_num+1}, Records: {len(X)}")
                
        except Exception as e:
            print(f"Error processing file {os.path.basename(csv_file)}: {str(e)}")
            continue
    
    # Membuat visualisasi feature importance
    if total_processed > 0:
        feature_importance = pd.DataFrame({
            'feature': expected_features,
            'importance': feature_importance_sum / total_processed
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(10), feature_importance['importance'][:10])
        plt.xticks(range(10), feature_importance['feature'][:10], rotation=45)
        plt.title('Top 10 Feature Importance (Averaged)')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, 'feature_importance.png'))
        plt.close()
        
        # Menyimpan informasi model
        model_info = {
            'Number of features': len(expected_features),
            'Total samples processed': total_processed,
            'Feature names': expected_features,
            'Model parameters': model.get_params(),
            'Files processed': csv_files
        }
        
        with open(os.path.join(reports_folder, 'model_info.txt'), 'w') as f:
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
        
        feature_importance.to_csv(os.path.join(reports_folder, 'feature_importance.csv'), index=False)
    
    print("\nTraining completed!")
    print(f"Total records processed: {total_processed}")
    print(f"Generated files in: {result_folder}")
    
    return result_folder

if __name__ == "__main__":
    data_folder = "B:/archive"
    print("Starting incremental training...")
    result_folder = train_incrementally(data_folder)