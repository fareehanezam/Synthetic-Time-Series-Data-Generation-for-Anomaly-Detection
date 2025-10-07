import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import get_logger
logger = get_logger("DataPreprocessing")

from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Configuration for preprocessing
SEQUENCE_LENGTH = 24 # 24 hours of data for each sequence
TARGET_CLASS = 'RECOVERING' # Class to generate synthetic data for

# --- Function to create sequences ---
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def preprocess_data(df, status_counts, sequence_length=SEQUENCE_LENGTH, target_class=TARGET_CLASS):
    logger.info("Starting data preprocessing for TimeGAN.")

    # Identify numerical sensor features (all columns except timestamp, machine_status, sensor_15)
    # Sensor 15 is categorical/object type, so we will drop it for simplicity.
    if 'sensor_15' in df.columns:
        df = df.drop(columns=['sensor_15'])
        logger.info("Dropped 'sensor_15' column as it is non-numeric.")

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    logger.info(f"Identified {len(numerical_cols)} numerical columns for scaling.")

    # Scale the numerical features
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    logger.info("Numerical features scaled using MinMaxScaler.")

    # Isolate data for the target minority class to train the GAN
    # We focus on 'RECOVERING' as 'BROKEN' has too few samples for a GAN to learn from.
    if status_counts.get('BROKEN', 0) < sequence_length * 2:
        logger.warning(f"'BROKEN' class has only {status_counts.get('BROKEN', 0)} samples. This is insufficient to train a GAN. Skipping TimeGAN for 'BROKEN'.")
        # If 'BROKEN' is insufficient, only use 'RECOVERING' data if it's sufficient
        if status_counts.get('RECOVERING', 0) < sequence_length * 2:
             logger.error(f"'RECOVERING' class has only {status_counts.get('RECOVERING', 0)} samples. This is insufficient to train a GAN. Cannot proceed with TimeGAN.")
             return None, None, None, None # Return None if no suitable data is available
        else:
            real_data_subset = df_scaled[df_scaled['machine_status'] == target_class][numerical_cols].values
            logger.info(f"Isolated data for '{target_class}' class. Shape: {real_data_subset.shape}")
    else:
        # If both 'BROKEN' and 'RECOVERING' were sufficient, you might choose to combine them or pick one
        # For this project's objective (synthesize 'RECOVERING'), we stick to TARGET_CLASS
        if status_counts.get(target_class, 0) < sequence_length * 2:
             logger.error(f"Target class '{target_class}' has only {status_counts.get(target_class, 0)} samples. This is insufficient to train a GAN. Cannot proceed with TimeGAN.")
             return None, None, None, None # Return None if target class is insufficient
        else:
            real_data_subset = df_scaled[df_scaled['machine_status'] == target_class][numerical_cols].values
            logger.info(f"Isolated data for '{target_class}' class. Shape: {real_data_subset.shape}")


    # Create sequences from the real data
    real_sequences = create_sequences(real_data_subset, sequence_length)
    logger.info(f"Created {len(real_sequences)} sequences of length {sequence_length} for '{target_class}' class.")

    print(f"Shape of the sequential data for TimeGAN training: {real_sequences.shape}")
    # Expected shape: (num_samples, sequence_length, num_features)

    return real_sequences, scaler, numerical_cols, df # Return scaler and numerical_cols for later use, and the potentially modified df
