import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, Input, Dropout
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
import numpy as np

class TimeGAN:
    def __init__(self, seq_len, n_features, hidden_dim, gamma=1, g_loss_v_weight=15.0):
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.g_loss_v_weight = g_loss_v_weight

        # Optimizers - Adjusted learning rates slightly
        self.g_opt = Adam(learning_rate=0.0002, beta_1=0.5) # Slightly increased G learning rate
        self.d_opt = Adam(learning_rate=00.0002, beta_1=0.5) # Slightly increased D learning rate
        self.ae_opt = Adam(learning_rate=0.00015) # Adjusted AE learning rate
        self.s_opt = Adam(learning_rate=0.00015) # Adjusted S learning rate

        # Losses
        self.bce = BinaryCrossentropy()
        self.mse = MeanSquaredError()

        # Build the models
        self._build_model()

    def _build_encoder(self):
        model = Sequential(name='Encoder')
        model.add(GRU(self.hidden_dim, return_sequences=True, input_shape=(self.seq_len, self.n_features)))
        model.add(GRU(self.hidden_dim, return_sequences=True))
        model.add(GRU(self.hidden_dim, return_sequences=True)) # Added another GRU layer
        model.add(Dropout(0.2))
        return model

    def _build_decoder(self):
        model = Sequential(name='Decoder')
        model.add(GRU(self.hidden_dim, return_sequences=True, input_shape=(self.seq_len, self.hidden_dim)))
        model.add(GRU(self.hidden_dim, return_sequences=True)) # Added another GRU layer
        model.add(Dense(self.n_features, activation='sigmoid'))
        return model

    def _build_generator(self):
        model = Sequential(name='Generator')
        model.add(GRU(self.hidden_dim, return_sequences=True, input_shape=(self.seq_len, self.n_features)))
        model.add(GRU(self.hidden_dim, return_sequences=True))
        model.add(GRU(self.hidden_dim, return_sequences=True)) # Added another GRU layer
        model.add(Dropout(0.2))
        return model

    def _build_supervisor(self):
        model = Sequential(name='Supervisor')
        model.add(GRU(self.hidden_dim, return_sequences=True, input_shape=(self.seq_len, self.hidden_dim)))
        model.add(GRU(self.hidden_dim, return_sequences=True)) # Added another GRU layer
        model.add(Dense(self.hidden_dim, activation='sigmoid'))
        return model

    def _build_discriminator(self):
        model = Sequential(name='Discriminator')
        # Increased complexity slightly by adding another layer
        model.add(GRU(self.hidden_dim, return_sequences=True, input_shape=(self.seq_len, self.hidden_dim)))
        model.add(GRU(self.hidden_dim, return_sequences=False))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def _build_model(self):
        # Component models
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.generator = self._build_generator()
        self.supervisor = self._build_supervisor()
        self.discriminator = self._build_discriminator()

        # --- Combined models for training ---
        # Autoencoder model
        x_input = Input(shape=(self.seq_len, self.n_features))
        self.autoencoder = Model(x_input, self.decoder(self.encoder(x_input)))


    @tf.function
    def train_autoencoder(self, X):
        with tf.GradientTape() as tape:
            X_tilde = self.autoencoder(X)
            e_loss = self.mse(X, X_tilde)
        grads = tape.gradient(e_loss, self.autoencoder.trainable_variables)
        self.ae_opt.apply_gradients(zip(grads, self.autoencoder.trainable_variables))
        return e_loss

    @tf.function
    def train_supervisor(self, H):
        with tf.GradientTape() as tape:
            H_hat_super = self.supervisor(H)
            # Supervised loss: predict next step in latent space
            g_loss_s = self.mse(H[:, 1:, :], H_hat_super[:, :-1, :])
        # Update supervisor weights
        grads = tape.gradient(g_loss_s, self.supervisor.trainable_variables)
        self.s_opt.apply_gradients(zip(grads, self.supervisor.trainable_variables))
        return g_loss_s

    @tf.function
    def train_generator(self, X, Z):
        with tf.GradientTape() as tape:
            # Generate synthetic latent sequence
            E_hat = self.generator(Z)
            H_hat = self.supervisor(E_hat)

            # Reconstruct synthetic data
            X_hat = self.decoder(E_hat) # Use E_hat, direct generator output

            # Discriminate synthetic latent sequence
            Y_fake = self.discriminator(H_hat)

            # --- Losses ---
            # 1. Adversarial loss
            g_loss_u = self.bce(tf.ones_like(Y_fake), Y_fake)

            # 2. Supervised loss (Corrected)
            g_loss_s = self.mse(E_hat[:, 1:, :], H_hat[:, :-1, :])

            # 3. Reconstruction loss (on synthetic data)
            g_loss_v = self.mse(X, X_hat)

            # Combine losses
            # Adjusted weights for combined loss
            # Using self.g_loss_v_weight from config/init
            g_loss = g_loss_u + 1.0 * tf.sqrt(g_loss_s) + self.g_loss_v_weight * g_loss_v

        # Update generator and supervisor weights jointly
        trainable_vars = self.generator.trainable_variables + self.supervisor.trainable_variables
        grads = tape.gradient(g_loss, trainable_vars)
        self.g_opt.apply_gradients(zip(grads, trainable_vars))
        return g_loss_u, g_loss_s, g_loss_v

    @tf.function
    def train_discriminator(self, X, Z):
        with tf.GradientTape() as tape:
            H = self.encoder(X) # Real latent sequence
            E_hat = self.generator(Z) # Fake latent sequence (direct from G)
            H_hat = self.supervisor(E_hat) # Fake latent sequence (supervised)

            Y_real = self.discriminator(H)
            Y_fake = self.discriminator(H_hat)

            d_loss_real = self.bce(tf.ones_like(Y_real), Y_real)
            d_loss_fake = self.bce(tf.zeros_like(Y_fake), Y_fake)

            d_loss = d_loss_real + d_loss_fake

        # Update discriminator weights
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return d_loss

    def generate(self, n_samples):
        Z = self.sample_noise(n_samples)
        E_hat = self.generator(Z)
        # Pass through decoder to get data in original feature space
        generated_data = self.decoder(E_hat)
        return generated_data.numpy()

    def sample_noise(self, n_samples):
        return np.random.uniform(size=[n_samples, self.seq_len, self.n_features])
