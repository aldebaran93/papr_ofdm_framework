import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import time

def ccdf_from_vals(papr_vals_db, bins=None):
    if bins is None:
        bins = np.linspace(0, np.max(papr_vals_db)+1, 200)
    ccdf = np.array([np.mean(papr_vals_db > b) for b in bins])
    return bins, ccdf

def plot_ccdf(channel, all_data: dict):
    plt.figure(figsize=(8,5))
    bins = np.linspace(0, 14, 300)
    for label, vals in all_data.items():
        bs, ccdf = ccdf_from_vals(vals, bins=bins)
        plt.semilogy(bs, ccdf, label=label)
    plt.xlabel('PAPR [dB]'); plt.ylabel('CCDF (Prob PAPR > PAPR0)')
    plt.grid(True, which='both'); plt.legend(); plt.tight_layout()
    plt.savefig(f"C:\\OFDM\\PAPR_OFDM\\papr_ofdm_framework\\figures\\PAPR_{channel}_{int(time.time())}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def plot_psd_signals(channel, signals: dict, fs=1.0, nfft=4096):
    plt.figure(figsize=(8,5))
    for label, x in signals.items():
        f, Pxx = welch(x, fs=fs, nperseg=min(1024,len(x)), nfft=nfft)
        Pdb = 10*np.log10(Pxx + 1e-12)
        plt.plot(f, Pdb, label=label)
    plt.xlabel('Normalized Frequency'); plt.ylabel('PSD [dB]'); plt.title('PSD Comparison'); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"C:\\OFDM\\PAPR_OFDM\\papr_ofdm_framework\\figures\\PSD_{channel}_{int(time.time())}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
