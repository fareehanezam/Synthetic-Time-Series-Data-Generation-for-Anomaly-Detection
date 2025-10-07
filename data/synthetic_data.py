
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory of src to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logger import get_logger # Import the logger initialization function
logger = get_logger("SyntheticDataGeneration") # Initialize the logger

def generate_synthetic_data(status_counts, trained_timegan_model, sequence_length, n_features, scaler, numerical_cols, target_class):
    """
    Generates synthetic time-series data using a trained TimeGAN model.

    Args:
        status_counts (pd.Series): Value counts of the 'machine_status' column.
        trained_timegan_model (TimeGAN): The trained TimeGAN model instance.
        sequence_length (int): The length of each time sequence.
        n_features (int): The number of features in each time step.
        scaler (MinMaxScaler): The scaler fitted on the real data.
        numerical_cols (list): List of numerical column names used for training.
        target_class (str): The target class for which synthetic data is generated.

    Returns:
        pd.DataFrame: DataFrame containing the generated synthetic data.
    """
    logger.info("Starting synthetic data generation.")

    # Define how many samples to generate
    # generate enough to make 'RECOVERING' have a similar count to 'NORMAL'
    n_normal = status_counts['NORMAL']
    n_recovering = status_counts['RECOVERING']
    n_to_generate = n_normal - n_recovering

    # We generate sequences, so we need to calculate how many sets of sequences to generate.
    # Ensures we generate a positive number of sequences
    if n_to_generate > 0:
        # Calculate the number of sequences needed. Each sequence has 'sequence_length' samples.
        # We need to generate enough sequences so that when flattened, we get at least n_to_generate samples.
        n_sequences_to_generate = (n_to_generate + sequence_length - 1) // sequence_length # Ceiling division
        logger.info(f"Generating {n_sequences_to_generate} synthetic sequences.")

        # Generate the data using the trained TimeGAN model
        synthetic_sequences = trained_timegan_model.generate(n_sequences_to_generate)
        print(f"Shape of generated synthetic sequences: {synthetic_sequences.shape}")

        # Inverse transform the data to its original scale
        synthetic_data_flat = synthetic_sequences.reshape(-1, n_features)
        synthetic_data_unscaled = scaler.inverse_transform(synthetic_data_flat)

        # Create a DataFrame for the synthetic data
        synthetic_df = pd.DataFrame(synthetic_data_unscaled, columns=numerical_cols)
        synthetic_df['machine_status'] = target_class # Assign the correct label

        # Trim to the exact number needed
        synthetic_df = synthetic_df.head(n_to_generate)
        logger.info(f"Generated and created DataFrame for {len(synthetic_df)} synthetic samples.")

        print("\nHead of the new synthetic data:")
        display(synthetic_df.head())
    else:
        logger.info("No synthetic data generation needed as minority class is not smaller than majority.")
        synthetic_df = pd.DataFrame() # Create empty dataframe if no generation is needed

    return synthetic_df

# This part will only run if the script is executed directly, not when imported by %run -i
if __name__ == "__main__":
    logger.warning("This script is intended to be run using %run -i from the notebook.")
    # Example usage if run directly (requires status_counts, trained_timegan_model, SEQUENCE_LENGTH, N_FEATURES, scaler, numerical_cols, TARGET_CLASS to be defined)
    # synthetic_df = generate_synthetic_data(status_counts, trained_timegan_model, SEQUENCE_LENGTH, N_FEATURES, scaler, numerical_cols, TARGET_CLASS)
    # print(synthetic_df)
