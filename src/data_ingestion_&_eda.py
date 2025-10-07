import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Configure plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

from src.logger import get_logger
logger = get_logger("DataIngestionEDA")

logger.info("Starting Data Ingestion and EDA.")

# Define results directory
results_dir = 'results/'
os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists

# Download the dataset
dataset_path = 'nphantawee/pump-sensor-data'
data_dir = 'data/'
zip_file_path = os.path.join(data_dir, 'pump-sensor-data.zip') # Name of the downloaded zip file
csv_file_path = os.path.join(data_dir, 'sensor.csv')

if not os.path.exists(csv_file_path):
    logger.info(f"Downloading dataset from Kaggle: {dataset_path}")
    # Use os.system to run the kaggle command from within the script
    download_command = f'kaggle datasets download -d {dataset_path} -p {data_dir} --unzip'
    os.system(download_command)
    logger.info("Dataset downloaded and unzipped.")
else:
    logger.info("Dataset already exists locally.")

# Load the dataset
try:
    df = pd.read_csv(csv_file_path)
    logger.info("Dataset loaded into pandas DataFrame.")
except FileNotFoundError:
    logger.error("sensor.csv not found in data directory after download attempt.")
    sys.exit("Execution stopped: sensor.csv not found.")

# Basic Data Inspection
logger.info(f"Dataset shape: {df.shape}")
print("Dataset Head:")
display(df.head())

# Drop the unnamed column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    logger.info("Dropped 'Unnamed: 0' column.")

# Convert timestamp to datetime object
df['timestamp'] = pd.to_datetime(df['timestamp'])
logger.info("Converted 'timestamp' column to datetime.")

# Check for missing values
logger.info("Checking for missing values.")
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Most sensor data has missing values. Using forward fill is a reasonable strategy for time-series data.
df.ffill(inplace=True)
df.bfill(inplace=True) # Backfill for any remaining NaNs at the beginning
logger.info("Missing values handled using forward and backward fill.")

# Analyze the target variable: machine_status
logger.info("Analyzing target variable 'machine_status'.")
status_counts = df['machine_status'].value_counts()
print("\nMachine Status Distribution:")
print(status_counts)

# Plot the distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='machine_status', data=df, order=status_counts.index)
plt.title('Distribution of Machine Status')
plt.xlabel('Machine Status')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.savefig(os.path.join(results_dir, 'machine_status_distribution.png'))
plt.show()

# --- Pie Chart for Machine Status ---
plt.figure(figsize=(10, 8))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140,
        colors=sns.color_palette('viridis', len(status_counts)))
plt.title('Proportional Distribution of Machine Status')
plt.ylabel('') # Hides the 'machine_status' label on the side
plt.savefig(os.path.join(results_dir, 'machine_status_pie_chart.png'))
plt.show()
print("\n")

# Select only the numeric sensor columns for correlation
sensor_cols = df.select_dtypes(include=np.number).columns
corr_matrix = df[sensor_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title('Correlation Matrix of Sensor Features')
plt.savefig(os.path.join(results_dir, 'sensor_correlation_heatmap.png'))
plt.show()

print("\n")
sensors_to_plot = ['sensor_01', 'sensor_02', 'sensor_06', 'sensor_07']

df[sensors_to_plot].hist(bins=50, figsize=(15, 10))
plt.suptitle('Distribution of Key Sensor Readings')
plt.savefig(os.path.join(results_dir, 'sensor_histograms.png'))
plt.show()
print("\n")

# --- Time-Series Plot for a Single Sensor ---
plt.figure(figsize=(18, 6))
sensor_to_plot_time = 'sensor_06'
sns.lineplot(x='timestamp', y=sensor_to_plot_time, data=df, alpha=0.8)
plt.title(f'Time-Series of {sensor_to_plot_time}')
plt.xlabel('Date')
plt.ylabel('Sensor Reading')
plt.savefig(os.path.join(results_dir, 'sensor_timeseries_plot.png'))
plt.show()

logger.info("EDA complete. The dataset is highly imbalanced.")
