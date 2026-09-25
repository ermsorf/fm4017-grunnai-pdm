from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

def preprocess_data(data_directory):
    all_data = []
    
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            data = pd.read_csv(file_path)
            all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)

    # Data cleaning: remove rows with NaN values
    cleaned_data = combined_data.dropna()

    # Normalization: scale features
    scaler = StandardScaler()
    feature_columns = cleaned_data.columns[:-1]  # Assuming last column is the target
    cleaned_data[feature_columns] = scaler.fit_transform(cleaned_data[feature_columns])

    return cleaned_data