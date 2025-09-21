"""
train_dl.py

Trains the OFDM autoencoder (dl_papr_module.OFDMAutoencoder) and compares it with
classical reducers. Produces CCDF, BER-vs-SNR, PSD plots for AWGN, Rayleigh, Rician channels.

"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# import your framework modules (assumes they are in same folder or installed)
from ofdm import OFDMSignal
from reducers import IdentityReducer, Clipping, Companding, SLM, PTS, ToneReservationCVX
from channel import awgn, rayleigh_fading, rician_fading, RappHPA
from analysis import ccdf_from_vals, plot_psd_signals

from dl_papr_module import OFDMAutoencoder

tf.get_logger().setLevel('ERROR')
# ----------------------------
# Experiment parameters
# ----------------------------
N = 64
L = 4
Nos = N * L
n_train_epochs = 30
steps_per_epoch = 200
batch_size = 128
snr_train = 12.0  # during training AWGN
alpha = 1.0
beta = 0.002

# output folder for figures
FIG_DIR = os.path.abspath("./figures_dl")
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------
# Create OFDM helper, reducers
# ----------------------------
ofdm = OFDMSignal(n_subcarriers=N, oversample=L)
hpa = RappHPA(A_sat=1.0, p=3.0)

reducers = {
    'Original': IdentityReducer(),
    'Clipping': Clipping(clipping_ratio=1.0),
    'Companding': Companding(mu=255),
    'SLM': SLM(n_candidates=8, si_bits=8),
    'PTS': PTS(V=4, phase_set=[1,-1,1j,-1j], search='greedy'),
    'ToneReserveCVX': ToneReservationCVX(reserved_tones=[0,1,N-2,N-1])
}

# ----------------------------
# Helper functions
# ----------------------------
def gen_qpsk_indices(batch_size, N):
    return np.random.randint(0, 4, size=(batch_size, N)).astype(np.int32)

def idx_to_complex_numpy(idx):
    # idx shape (B,N)
    angles = (np.pi/4) + (np.pi/2) * idx
    return np.exp(1j * angles)

# detect via FFT and nearest QPSK angle
def qpsk_decision_from_freq(X_hat):
    angles = np.angle(X_hat)
    angles = np.mod(angles, 2*np.pi)
    const_angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
    diffs = np.abs(angles[..., None] - const_angles[None, ...])
    diffs = np.minimum(diffs, 2*np.pi - diffs)
    idx = np.argmin(diffs, axis=-1)
    return idx  # (B,N)

# simple PAPR db
def papr_db_of_signal(x):
    p = np.abs(x)**2
    return 10*np.log10(np.max(p, axis=1) / (np.mean(p, axis=1)+1e-12))

# ----------------------------
# Create and compile model
# ----------------------------
model = OFDMAutoencoder(N=N, oversample=L, enc_hidden=512, dec_hidden=512, alpha=alpha, beta=beta)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# training loop (eager)
print("Start training autoencoder...")

for epoch in range(n_train_epochs):
    epoch_loss = 0.0
    epoch_ce = 0.0
    epoch_papr = 0.0
    for step in range(steps_per_epoch):
        batch_idx = gen_qpsk_indices(batch_size, N)
        with tf.GradientTape() as tape:
            logits, x_time, x_enc, papr_db = model(batch_idx, training=True, snr_db=snr_train, channel='awgn', apply_hpa=False)
            loss, ce_mean, papr_loss = model.compute_losses(logits, batch_idx, papr_db)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += float(loss.numpy())
        epoch_ce += float(ce_mean.numpy())
        epoch_papr += float(np.mean(papr_db.numpy()))
    epoch_loss /= steps_per_epoch
    epoch_ce /= steps_per_epoch
    epoch_papr /= steps_per_epoch
    print(f"Epoch {epoch+1}/{n_train_epochs} - loss {epoch_loss:.4f}, CE {epoch_ce:.4f}, avg PAPR(dB) {epoch_papr:.3f}")

# save weights
ts = int(time.time())
model_save = f"dl_autoenc_{ts}.weights.h5"
model.save_weights(model_save)
print("Model saved:", model_save)

# ----------------------------
# Evaluation: CCDF, BER, PSD for three channels
# ----------------------------
channels = {
    'awgn': lambda x, snr: awgn(x, snr),
    'rayleigh': lambda x, snr: awgn(rayleigh_fading(x), snr),
    'rician': lambda x, snr: awgn(rician_fading(x), snr)
}

snr_list = [0,5,10,15,20]
n_eval = 500  # OFDM symbols per reducer/channel for CCDF and BER

results = {}

for ch_name, ch_fn in channels.items():
    print(f"\n--- Evaluating channel: {ch_name} ---")
    results[ch_name] = {}
    # CCDF: collect papr arrays per reducer
    papr_collections = {}
    for rname, reducer in reducers.items():
        papr_vals = []
        # for DL model, we generate encoded time signals using TF model.predict
        for i in range(n_eval):
            idx = gen_qpsk_indices(1, N)
            if rname == 'DL_AE':
                logits, x_time, x_enc, papr_db = model(idx, training=False, snr_db=12.0, channel='awgn', apply_hpa=False)
                x_out = x_enc.numpy().reshape(-1)
            else:
                X = idx_to_complex_numpy(idx).reshape(N)
                x_out, meta = reducer.apply(ofdm, X)
            papr_vals.append(10*np.log10(np.max(np.abs(x_out)**2) / (np.mean(np.abs(x_out)**2)+1e-12)))
        papr_collections[rname] = np.array(papr_vals)

    # add DL model papr if not in reducers
    # generate DL papr only once and insert under 'DL_AE'
    dl_papr_vals = []
    for i in range(n_eval):
        idx = gen_qpsk_indices(1, N)
        logits, x_time, x_enc, papr_db = model(idx, training=False, snr_db=12.0, channel='awgn', apply_hpa=False)
        dl_papr_vals.append(float(papr_db.numpy()))
    papr_collections['DL_AE'] = np.array(dl_papr_vals)

    # Plot CCDF for this channel
    plt.figure(figsize=(7,4))
    bins = np.linspace(0, 14, 300)
    for label, vals in papr_collections.items():
        bs, ccdf = ccdf_from_vals(vals, bins=bins)
        plt.semilogy(bs, ccdf, label=label)
    plt.xlabel('PAPR [dB]'); plt.ylabel('CCDF'); plt.title(f'CCDF - {ch_name}')
    plt.grid(True); plt.legend(); plt.tight_layout()
    fname = os.path.join(FIG_DIR, f"CCDF_{ch_name}_{ts}.pdf")
    plt.savefig(fname, bbox_inches='tight')
    print("Saved CCDF to", fname)
    plt.close()

    # BER vs SNR for each reducer (including DL)
    ber_results = {}
    for rname, reducer in list(reducers.items()) + [('DL_AE', None)]:
        ber_list = []
        for snr in snr_list:
            bit_err = 0; total_bits = 0
            for i in range(n_eval):
                idx = gen_qpsk_indices(1, N).reshape(N)
                # transmitter
                if rname == 'DL_AE':
                    logits, x_time, x_enc_complex, papr_db = model(idx.reshape(1,N), training=False, snr_db=snr, channel='none', apply_hpa=False)
                    x_tx = x_enc_complex.numpy().reshape(-1)
                else:
                    X = idx_to_complex_numpy(idx)
                    x_tx, meta = reducer.apply(ofdm, X)
                # apply HPA
                x_pa = hpa.apply(x_tx)
                # channel
                y = ch_fn(x_pa, snr)
                # receiver (FFT)
                X_hat = ofdm.ofdm_fft(y)
                decided_idx = qpsk_decision_from_freq(X_hat) if isinstance(decided_idx:=None, type(None)) else None
                decided_idx = qpsk_decision_from_freq(X_hat)
                # error counting
                bit_err += np.sum(decided_idx != idx)
                total_bits += N
            ser = bit_err / total_bits
            ber = ser / 2.0  # QPSK: 2 bits per symbol
            ber_list.append(ber)
            print(f"{rname} ({ch_name}) SNR={snr} dB -> BER={ber:.3e}")
        ber_results[rname] = np.array(ber_list)

    # Plot BER vs SNR for this channel
    plt.figure(figsize=(7,4))
    for label, ber_vals in ber_results.items():
        plt.semilogy(snr_list, ber_vals, marker='o', label=label)
    plt.xlabel('SNR (dB)'); plt.ylabel('BER'); plt.title(f'BER vs SNR - {ch_name}')
    plt.grid(True); plt.legend(); plt.tight_layout()
    fname = os.path.join(FIG_DIR, f"BER_{ch_name}_{ts}.pdf")
    plt.savefig(fname, bbox_inches='tight')
    print("Saved BER plot to", fname)
    plt.close()

    # PSD: choose a few signals (DL and Original) and plot PSD before/after HPA
    # gather signals
    signals = {}
    # 50 blocks concatenated
    for choose in ['Original', 'DL_AE']:
        collected = []
        for i in range(50):
            idx = gen_qpsk_indices(1, N).reshape(N)
            if choose == 'DL_AE':
                _, _, x_enc, _ = model(idx.reshape(1,N), training=False, snr_db=snr_train, channel='none', apply_hpa=False)
                x = x_enc.numpy().reshape(-1)
            else:
                X = idx_to_complex_numpy(idx)
                x, _ = reducers['Original'].apply(ofdm, X)
            collected.append(x)
        x_concat = np.concatenate(collected)
        signals[f"{choose}_beforeHPA"] = x_concat
        signals[f"{choose}_afterHPA"] = hpa.apply(x_concat)
    # use your plot_psd_signals
    plot_psd_signals("awgn", signals)
    fname = os.path.join(FIG_DIR, f"PSD_{ch_name}_{ts}.pdf")
    plt.savefig(fname, bbox_inches='tight')  # some plotting functions already called show; try to save last fig
    print("Saved PSD to", fname)
    plt.close()

print("All evaluations complete. Figures saved under:", FIG_DIR)
