"""
dl_papr_module.py

Autoencoder-style OFDM transmitter (encoder) + decoder for PAPR reduction.
Designed to integrate with my OFDM framework (OFDMSignal, channel functions, reducers).
"""

import numpy as np
import tensorflow as tf

# -- Utilities
def complex_to_2ch_tf(x):
    return tf.cast(tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1), tf.float32)
def ch2complex_tf(x):
    return tf.cast(x[..., 0], tf.complex64) + 1j * tf.cast(x[..., 1], tf.complex64)

def qpsk_indices_to_complex_tf(idx):
    # idx: (B, N) ints 0..3
    angles = (np.pi/4) + (np.pi/2) * tf.cast(idx, tf.float32)
    return tf.cast(tf.math.cos(angles), tf.complex64) + 1j * tf.cast(tf.math.sin(angles), tf.complex64)

# -- Channel models inside TF (lightweight)
def rapp_hpa_tf(x, A_sat=1.0, p=3.0):
    mag = tf.abs(x)
    denom = tf.pow(1.0 + tf.pow(mag / (A_sat + 1e-12), 2.0 * p), 1.0 / (2.0 * p))
    return x / denom

def awgn_tf(signal, snr_db):
    sig_power = tf.reduce_mean(tf.abs(signal)**2, axis=-1, keepdims=True)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = sig_power / (snr_lin + 1e-12)
    sigma = tf.expand_dims(tf.sqrt(noise_var / 2.0), axis=-1)
    #sigma = tf.sqrt(noise_var / 2.0)
    noise = tf.complex(tf.random.normal(tf.shape(signal), stddev=sigma[...,0]),
                       tf.random.normal(tf.shape(signal), stddev=sigma[...,0]))
    return signal + noise

# -- The Keras model: encoder + decoder
class OFDMAutoencoder(tf.keras.Model):
    def __init__(self, N=64, oversample=4, enc_hidden=256, dec_hidden=256, alpha=1.0, beta=0.002):
        super().__init__()
        self.N = N
        self.L = oversample
        self.Nos = N * oversample
        self.alpha = alpha
        self.beta = beta

        # Encoder network (operates on time-domain real/imag flattened)
        self.enc_fc1 = tf.keras.layers.Dense(enc_hidden, activation='relu')
        self.enc_fc2 = tf.keras.layers.Dense(enc_hidden, activation='relu')
        self.enc_out = tf.keras.layers.Dense(self.Nos * 2, activation=None)

        # Decoder network (operates on frequency-domain real/imag flattened)
        self.dec_fc1 = tf.keras.layers.Dense(dec_hidden, activation='relu')
        self.dec_fc2 = tf.keras.layers.Dense(dec_hidden, activation='relu')
        self.dec_out = tf.keras.layers.Dense(self.N * 4, activation=None)  # logits per subcarrier (4 QPSK classes)

    def call(self, idx_batch, training=False, snr_db=15.0, channel='awgn', apply_hpa=False):
        """
        idx_batch: integer indices (B, N) of QPSK symbols
        returns (logits (B,N,4), x_time (B, Nos) original, x_enc_time (B, Nos) encoded complex, papr_db (B,))
        """
        idx = tf.cast(idx_batch, tf.int32)
        # map to complex symbols
        X = qpsk_indices_to_complex_tf(idx)  # (B,N)

        # oversample into X_os length Nos
        half = self.N // 2
        batch = tf.shape(X)[0]
        zeros_mid = tf.zeros((batch, self.Nos - 2*half), dtype=tf.complex64)
        left = X[:, :half]
        right = X[:, half:]
        X_os = tf.concat([left, zeros_mid, right], axis=1)  # (B, Nos)

        # IFFT -> time domain (TF op)
        x_time = tf.signal.ifft(X_os) * tf.cast(self.L, tf.complex64)  # (B, Nos)

        # Encoder: produce additive delta in real/imag domain
        x_2ch = complex_to_2ch_tf(x_time)  # (B, Nos, 2)
        x_flat = tf.reshape(x_2ch, [batch, self.Nos * 2])
        h = self.enc_fc1(x_flat)
        h = self.enc_fc2(h)
        delta = self.enc_out(h)
        delta = tf.reshape(delta, [batch, self.Nos, 2])
        x_enc_2ch = x_2ch + delta
        x_enc_cplx = ch2complex_tf(x_enc_2ch)  # (B, Nos)

        # PAPR compute (smooth approx)
        power = tf.abs(x_enc_cplx)**2
        temp = 0.1
        power_ls = tf.reduce_logsumexp(power / temp, axis=1) * temp  # approx max
        p_max = power_ls
        p_mean = tf.reduce_mean(power, axis=1) + 1e-12
        papr = p_max / p_mean
        papr_db = 10.0 * tf.math.log(papr + 1e-12) / tf.math.log(10.0)

        # optional HPA
        x_after_hpa = rapp_hpa_tf(x_enc_cplx) if apply_hpa else x_enc_cplx

        # Channel
        if channel == 'awgn':
            y = awgn_tf(x_after_hpa, snr_db)
        elif channel == 'none':
            y = x_after_hpa
        else:
            # user may want to call external (numpy) fading; for training we use awgn only or implement small fading here
            y = awgn_tf(x_after_hpa, snr_db)

        # FFT back -> frequency domain
        Y_os = tf.signal.fft(y) / tf.cast(self.L, tf.complex64)
        left_Y = Y_os[:, :half]
        right_Y = Y_os[:, -half:]
        Y = tf.concat([left_Y, right_Y], axis=1)  # (B,N)

        Y_2ch = complex_to_2ch_tf(Y)
        Y_flat = tf.reshape(Y_2ch, [batch, self.N * 2])

        d = self.dec_fc1(Y_flat)
        d = self.dec_fc2(d)
        logits = self.dec_out(d)
        logits = tf.reshape(logits, [batch, self.N, 4])

        return logits, x_time, x_enc_cplx, papr_db

    def compute_losses(self, logits, labels, papr_db):
        # cross-entropy
        labels_flat = tf.reshape(labels, [-1])
        logits_flat = tf.reshape(logits, [-1, 4])
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_flat, logits=logits_flat)
        ce_mean = tf.reduce_mean(ce)
        papr_lin = tf.reduce_mean(10.0**(papr_db/10.0))  # linear average
        papr_loss = papr_lin
        total = self.alpha * ce_mean + self.beta * papr_loss
        return total, ce_mean, papr_loss
