import numpy as np

class OFDMSignal:
    def __init__(self, n_subcarriers=64, oversample=4, cp_len=0):
        self.N = n_subcarriers
        self.os_factor = oversample
        self.Nos = self.N * self.os_factor
        self.cp_len = cp_len

    def random_qpsk_symbols(self, n_symbols: int):
        idx = np.random.randint(0, 4, size=(n_symbols, self.N))
        s = np.exp(1j * (np.pi/4 + (np.pi/2) * idx))
        return s, idx

    def ofdm_ifft(self, freq_data: np.ndarray):
        single = False
        if freq_data.ndim == 1:
            freq_data = freq_data[np.newaxis, :]
            single = True
        n_sym = freq_data.shape[0]
        X_os = np.zeros((n_sym, self.Nos), dtype=complex)
        half = self.N // 2
        X_os[:, :half] = freq_data[:, :half]
        X_os[:, -half:] = freq_data[:, half:]
        x_time = np.fft.ifft(X_os, axis=1) * self.os_factor
        if single:
            return x_time[0]
        return x_time

    def ofdm_fft(self, time_data: np.ndarray):
        single = False
        if time_data.ndim == 1:
            time_data = time_data[np.newaxis, :]
            single = True
        X_os = np.fft.fft(time_data, axis=1) / self.os_factor
        half = self.N // 2
        X = np.zeros((X_os.shape[0], self.N), dtype=complex)
        X[:, :half] = X_os[:, :half]
        X[:, half:] = X_os[:, -half:]
        if single:
            return X[0]
        return X

    def add_cp(self, x_time: np.ndarray):
        if self.cp_len <= 0:
            return x_time
        if x_time.ndim == 1:
            return np.concatenate((x_time[-self.cp_len:], x_time))
        else:
            return np.hstack((x_time[:, -self.cp_len:], x_time))
