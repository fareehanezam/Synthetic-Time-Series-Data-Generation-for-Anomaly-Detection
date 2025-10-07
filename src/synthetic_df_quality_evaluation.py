
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# Add the parent directory of src to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logger import get_logger # Import the logger initialization function
logger = get_logger("SyntheticDataEvaluation") # Initialize the logger

def evaluate_synthetic_data_quality(df, synthetic_df, numerical_cols, target_class='RECOVERING'):
    logger.info("Evaluating synthetic data quality with t-SNE.")

    # Prepare data for t-SNE
    # To make it computationally feasible, we take a sample of the 'NORMAL' data
    real_recovering_data = df[df['machine_status'] == target_class][numerical_cols]

    # Ensure we have enough data for sampling
    min_samples = min(len(real_recovering_data), len(synthetic_df))

    if min_samples == 0:
        logger.warning("Not enough data for t-SNE plot.")
        print("Not enough data for t-SNE plot.")
        return

    normal_sample = df[df['machine_status'] == 'NORMAL'][numerical_cols].sample(n=min_samples, random_state=42)
    synthetic_recovering_data = synthetic_df[numerical_cols].sample(n=min_samples, random_state=42) # Match sizes for fair comparison

    combined_data = pd.concat([
        normal_sample,
        real_recovering_data,
        synthetic_recovering_data
    ])

    labels = ['Normal (Real)'] * len(normal_sample) + \
             [f'{target_class} (Real)'] * len(real_recovering_data) + \
             [f'{target_class} (Synthetic)'] * len(synthetic_recovering_data)

    # Scale the combined data before applying t-SNE
    tsne_scaler = MinMaxScaler()
    combined_data_scaled = tsne_scaler.fit_transform(combined_data)

    # Apply t-SNE
    # Adjust perplexity and n_iter if needed for better visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
    tsne_results = tsne.fit_transform(combined_data_scaled)
    logger.info("t-SNE fitting complete.")

    # Plot the results
    tsne_df = pd.DataFrame({
        'tsne-2d-one': tsne_results[:,0],
        'tsne-2d-two': tsne_results[:,1],
        'label': labels
    })

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=sns.color_palette("hls", 3),
        data=tsne_df,
        legend="full",
        alpha=0.6
    )
    plt.title('t-SNE Plot of Real vs. Synthetic Data')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best')
    plt.savefig('results/tsne_real_vs_synthetic.png')
    plt.show()

    logger.info("t-SNE plot saved.")

# This part will only run if the script is executed directly, not when imported by %run -i
if __name__ == "__main__":
    logger.warning("This script is intended to be run by calling its function from the notebook.")
    # Example usage if run directly (requires df, synthetic_df, numerical_cols to be defined)
    # evaluate_synthetic_data_quality(df, synthetic_df, numerical_cols)
