import numpy as np
from itertools import product
from typing import List, Tuple, Dict
from scipy.fft import fft, ifft
import cvxpy as cp

def papr_db(signal: np.ndarray) -> float:
    power = np.abs(signal) ** 2
    return 10.0 * np.log10(np.max(power) / np.mean(power) + 1e-12)

class PAPRReducer:
    def apply(self, ofdm, freq_symbol: np.ndarray) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError

class IdentityReducer(PAPRReducer):
    def apply(self, ofdm, freq_symbol):
        return ofdm.ofdm_ifft(freq_symbol), {}

class Clipping(PAPRReducer):
    def __init__(self, clipping_ratio=1.0):
        self.cr = clipping_ratio
    def apply(self, ofdm, freq_symbol):
        x = ofdm.ofdm_ifft(freq_symbol)
        rms = np.sqrt(np.mean(np.abs(x)**2))
        A = self.cr * rms
        mag = np.abs(x)
        clipped = np.where(mag <= A, x, A * x / (mag + 1e-12))
        return clipped, {'threshold':A}

class Companding(PAPRReducer):
    def __init__(self, mu=255):
        self.mu = mu
    def mu_compand(self, x):
        max_val = np.max(np.abs(x))
        if max_val == 0:
            return x
        x_norm = x / max_val
        mag = np.abs(x_norm)
        comp_mag = np.log(1 + self.mu * mag) / np.log(1 + self.mu)
        return comp_mag * (x_norm / (mag + 1e-12)) * max_val
    def apply(self, ofdm, freq_symbol):
        x = ofdm.ofdm_ifft(freq_symbol)
        return self.mu_compand(x), {'mu': self.mu}

class SLM(PAPRReducer):
    def __init__(self, n_candidates=8, si_bits=8):
        self.U = n_candidates
        self.si_bits = si_bits  # number of side-information bits to represent selected candidate (if needed)
    def apply(self, ofdm, freq_symbol):
        best_x = None; best_p = 1e9; best_idx = 0; best_phase = None
        N = freq_symbol.size
        for u in range(self.U):
            phases = np.exp(1j*2*np.pi*np.random.rand(N))
            Xc = freq_symbol * phases
            x = ofdm.ofdm_ifft(Xc)
            p = papr_db(x)
            if p < best_p:
                best_p = p; best_x = x; best_phase = phases; best_idx = u
        # side info is index u (0..U-1)
        return best_x, {'best_papr_db': best_p, 'si_idx': int(best_idx), 'phase_vec': best_phase, 'si_bits': self.si_bits, 'U': self.U}

class PTS(PAPRReducer):
    def __init__(self, V=4, phase_set=None, search='greedy', si_bits=None):
        self.V = V
        self.phase_set = phase_set if phase_set is not None else [1, -1, 1j, -1j]
        self.search = search
        self.si_bits = si_bits if si_bits is not None else int(np.ceil(np.log2(len(self.phase_set)**self.V)))
    def partition(self, X):
        N = X.size; blocks=[]
        block_len = N//self.V
        for v in range(self.V):
            s = v*block_len; e = (v+1)*block_len if v!=self.V-1 else N
            b = np.zeros_like(X); b[s:e]=X[s:e]; blocks.append(b)
        return blocks
    def apply(self, ofdm, freq_symbol):
        blocks = self.partition(freq_symbol)
        x_blocks = [ofdm.ofdm_ifft(b) for b in blocks]
        best_p=1e9; best_x=None; best_phases=None; best_idx=None
        if self.search=='exhaustive':
            for idxs in product(self.phase_set, repeat=self.V):
                xsum = sum(idxs[v]*x_blocks[v] for v in range(self.V))
                p = papr_db(xsum)
                if p<best_p:
                    best_p=p; best_x=xsum; best_phases=idxs
        else:
            phases_chosen = [1]*self.V
            best_x = sum(phases_chosen[v]*x_blocks[v] for v in range(self.V))
            best_p = papr_db(best_x)
            for v in range(self.V):
                for phase in self.phase_set:
                    trial = phases_chosen.copy(); trial[v]=phase
                    xtrial = sum(trial[k]*x_blocks[k] for k in range(self.V))
                    p = papr_db(xtrial)
                    if p<best_p:
                        best_p=p; phases_chosen[v]=phase; best_x=xtrial
            best_phases = phases_chosen
        return best_x, {'best_papr_db': best_p, 'phases': best_phases, 'si_bits': self.si_bits}

class ToneReservationCVX(PAPRReducer):
    def __init__(self, reserved_tones: List[int], rho=1e-3):
        self.reserved = reserved_tones
        self.rho = rho

    def apply(self, ofdm, freq_symbol):
        x = ofdm.ofdm_ifft(freq_symbol)
        Nos = ofdm.Nos
        R = len(self.reserved)

        # cvx variables (real and imag for each reserved tone)
        c_re = cp.Variable(R)
        c_im = cp.Variable(R)

        # build mapping matrices
        A_re = np.zeros((Nos, R))
        A_im = np.zeros((Nos, R))
        n = np.arange(Nos)
        for i, k in enumerate(self.reserved):
            idx = k if k < ofdm.N//2 else k - ofdm.N + Nos
            basis = np.exp(1j*2*np.pi*idx*n / Nos)
            A_re[:, i] = np.real(basis)
            A_im[:, i] = np.imag(basis)

        # time-domain real/imag
        x_re = np.real(x)
        x_im = np.imag(x)

        # corrected signals as cvx expressions
        x_re_corr = x_re + A_re @ c_re - A_im @ c_im
        x_im_corr = x_im + A_re @ c_im + A_im @ c_re

        # define peak constraint
        t = cp.Variable()
        #constraints = [cp.abs(cp.sqrt(cp.square(x_re_corr) + cp.square(x_im_corr))) <= t]
        #constraints = [cp.sqrt(x_re_corr**2 + x_im_corr**2) <= t]
        constraints = [cp.norm(cp.hstack([x_re_corr, x_im_corr]), 2) <= t]

        # minimize peak + small regularization
        prob = cp.Problem(cp.Minimize(t + self.rho*(cp.norm(c_re,2)+cp.norm(c_im,2))), constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        # flatten solution to ensure proper broadcasting
        c_re_val = np.array(c_re.value).flatten()
        c_im_val = np.array(c_im.value).flatten()

        # apply correction in time domain
        x_re_corr_val = x_re + A_re @ c_re_val - A_im @ c_im_val
        x_im_corr_val = x_im + A_re @ c_im_val + A_im @ c_re_val
        x_tr = x_re_corr_val + 1j * x_im_corr_val

        return x_tr, {'initial_papr_db': papr_db(x), 'final_papr_db': papr_db(x_tr)}