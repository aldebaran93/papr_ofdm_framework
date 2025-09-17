
'''
import numpy as np
import matplotlib.pyplot as plt

# Example OFDM parameters
N = 64  # number of subcarriers
M = 16  # 16-QAM

# Generate random 16-QAM symbols
def qam16_symbols(num):
    re = 2*(np.random.randint(0, 4, num))-3
    im = 2*(np.random.randint(0, 4, num))-3
    return re + 1j*im

X = qam16_symbols(N)

# SLM phase vectors
U = 4  # number of candidate phase sequences
phase_vectors = np.exp(1j * 2 * np.pi * np.random.rand(U, N))

# Generate candidate time-domain OFDM symbols
ofdm_candidates = np.array([np.fft.ifft(X * pv) for pv in phase_vectors])

# Compute PAPR for each candidate
def papr(x):
    return 10*np.log10(np.max(np.abs(x)**2) / np.mean(np.abs(x)**2))

papr_values = np.array([papr(c) for c in ofdm_candidates])
best_idx = np.argmin(papr_values)
ofdm_tx = ofdm_candidates[best_idx]

# Plot original constellation vs transmitted SLM constellation
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(X.real, X.imag)
plt.title("Original QAM Constellation")
plt.grid(True)
plt.axis('equal')

plt.subplot(1,2,2)
plt.scatter(np.fft.fft(ofdm_tx).real, np.fft.fft(ofdm_tx).imag)
plt.title(f"SLM Transmitted Constellation (candidate {best_idx})")
plt.grid(True)
plt.axis('equal')

plt.show()

# Print PAPR values
print("PAPR values for candidates:", papr_values)
print("Selected candidate index:", best_idx)

'''