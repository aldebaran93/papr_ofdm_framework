import numpy as np

class RappHPA:
    def __init__(self, A_sat=1.0, p=3.0):
        self.A = A_sat
        self.p = p
    def apply(self, x: np.ndarray):
        mag = np.abs(x)
        denom = (1 + (mag / (self.A + 1e-12)) ** (2 * self.p)) ** (1.0 / (2 * self.p))
        return x / denom

def awgn(signal, snr_db):
    sig_power = np.mean(np.abs(signal)**2)
    snr_lin = 10**(snr_db/10.0)
    noise_var = sig_power / snr_lin
    noise = np.sqrt(noise_var/2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise

def bpsk_awgn_tx(bits, snr_db):
    s = 1 - 2*bits
    sig_power = np.mean(np.abs(s)**2)
    snr_lin = 10**(snr_db/10.0)
    noise_var = sig_power / snr_lin
    noise = np.sqrt(noise_var/2) * np.random.randn(*s.shape)
    r = s + noise
    decided = (r < 0).astype(int)
    return decided

# ======================
# NEW: Fading channels
# ======================

def rayleigh_fading(signal, n_taps=8, exp_decay=0.5):
    """
    Apply frequency-selective Rayleigh fading channel.
    - n_taps: number of multipath components
    - exp_decay: exponential power decay factor
    """
    if n_taps <= 0:
        n_taps=1
    # Random complex Gaussian taps with exponential decay
    h = (np.random.randn(n_taps) + 1j*np.random.randn(n_taps)) / np.sqrt(2)
    power = np.exp(-exp_decay * np.arange(n_taps))  # exponential decay
    h = h * np.sqrt(power / np.sum(power))          # normalize power
    faded = np.convolve(signal, h, mode="same")     # apply channel
    return faded

def rician_fading(signal, K=10, n_taps=8, exp_decay=0.5):
    """
    Apply frequency-selective Rician fading channel.
    - K: Rician factor (ratio LOS/NLOS)
    """
    if n_taps <= 0:
        n_taps=1
    # LOS component (real + deterministic phase)
    los = np.sqrt(K / (K+1))
    # NLOS multipath
    h_nlos = (np.random.randn(n_taps) + 1j*np.random.randn(n_taps)) / np.sqrt(2*(K+1))
    power = np.exp(-exp_decay * np.arange(n_taps))
    h_nlos = h_nlos * np.sqrt(power / np.sum(power))
    h = np.zeros_like(h_nlos)
    h[0] = los + h_nlos[0]  # first tap has LOS + fading
    h[1:] = h_nlos[1:]
    faded = np.convolve(signal, h, mode="same")
    return faded

def fading_awgn(signal, snr_db, fading, **kwargs):
    """
    Fading + AWGN combined
    - fading: "rayleigh" or "rician"
    - kwargs: fading parameters
    """
    if fading == "rayleigh":
        faded, h = rayleigh_fading(signal, **kwargs)
    elif fading == "rician":
        faded, h = rician_fading(signal, **kwargs)
    else:
        raise ValueError("Unsupported fading type")
    noisy = awgn(faded, snr_db)
    return noisy
