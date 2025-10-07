
import tensorflow as tf
from tqdm import tqdm
import sys
import os
import numpy as np

# Add the parent directory of src to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.timegan import TimeGAN
from src.logger import get_logger # Ensure logger is imported

logger = get_logger("TimeGAN_Training") # Initialize logger at the beginning

def train_timegan(real_sequences, sequence_length, n_features, hidden_dim=24, epochs=500, batch_size=128):
    """
    Trains the TimeGAN model.

    Args:
        real_sequences (np.ndarray): The real time-series data sequences for training.
        sequence_length (int): The length of each time sequence.
        n_features (int): The number of features in each time step.
        hidden_dim (int): The dimension of the hidden states in GRUs.
        epochs (int): The number of training epochs.
        batch_size (int): The batch size for training.

    Returns:
        TimeGAN: The trained TimeGAN model instance.
    """
    logger.info("Starting TimeGAN model training.")

    # Instantiate the TimeGAN model
    timegan = TimeGAN(seq_len=sequence_length, n_features=n_features, hidden_dim=hidden_dim)
    logger.info("TimeGAN model instantiated.")

    # --- 1. Autoencoder Training ---
    logger.info("Phase 1: Training Autoencoder.")
    for epoch in tqdm(range(epochs)):
        real_batch = tf.convert_to_tensor(real_sequences[np.random.permutation(len(real_sequences))[:batch_size]], dtype=tf.float32)
        loss = timegan.train_autoencoder(real_batch)
        if epoch % 100 == 0:
            print(f'Autoencoder Epoch {epoch}, Loss: {loss.numpy():.4f}')

    # --- 2. Supervisor Training ---
    logger.info("Phase 2: Training Supervisor.")
    for epoch in tqdm(range(epochs)):
        real_batch = tf.convert_to_tensor(real_sequences[np.random.permutation(len(real_sequences))[:batch_size]], dtype=tf.float32)
        H_real = timegan.encoder(real_batch)
        loss = timegan.train_supervisor(H_real)
        if epoch % 100 == 0:
            print(f'Supervisor Epoch {epoch}, Loss: {loss.numpy():.4f}')

    # --- 3. Joint Training (Generator and Discriminator) ---
    logger.info("Phase 3: Jointly Training Generator and Discriminator.")
    for epoch in tqdm(range(epochs)):
        # Train Generator
        real_batch = tf.convert_to_tensor(real_sequences[np.random.permutation(len(real_sequences))[:batch_size]], dtype=tf.float32)
        noise_batch = timegan.sample_noise(batch_size)
        g_loss_u, g_loss_s, g_loss_v = timegan.train_generator(real_batch, noise_batch)

        # Train Discriminator
        real_batch = tf.convert_to_tensor(real_sequences[np.random.permutation(len(real_sequences))[:batch_size]], dtype=tf.float32)
        noise_batch = timegan.sample_noise(batch_size)
        d_loss = timegan.train_discriminator(real_batch, noise_batch)

        if epoch % 100 == 0:
            print(f'Joint Epoch {epoch}, D Loss: {d_loss.numpy():.4f}, G Loss (Adv): {g_loss_u.numpy():.4f}, G Loss (Sup): {g_loss_s.numpy():.4f}, G Loss (Rec): {g_loss_v.numpy():.4f}')

    # Save the trained models
    timegan.generator.save('models/timegan_generator.h5')
    timegan.decoder.save('models/timegan_decoder.h5')
    logger.info("TimeGAN training complete and models saved.")
    return timegan
