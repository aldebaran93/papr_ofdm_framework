import numpy as np
from ofdm import OFDMSignal
from reducers import *
from channel import RappHPA, awgn, bpsk_awgn_tx, fading_awgn
from analysis import plot_ccdf, plot_psd_signals
import matplotlib.pyplot as plt
import time
import argparse

# QPSK decision helper
def qpsk_decision(symbols):
    angles = np.angle(symbols)
    angles = np.mod(angles, 2*np.pi)
    const_angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
    diffs = np.abs(angles[:, None] - const_angles[None, :])
    diffs = np.minimum(diffs, 2*np.pi - diffs)
    idx = np.argmin(diffs, axis=1)
    return idx

# Convert QPSK symbol indices to bits
def symbol_indices_to_bits(idx):
    mapping = {0:(0,0), 1:(0,1), 2:(1,1), 3:(1,0)}
    bits = np.array([mapping[i] for i in idx]).reshape(-1,2)
    return bits.reshape(-1)

class Experiment:
    def __init__(self, N=64, os_factor=4, n_symbols=1200, seed=0):
        np.random.seed(seed)
        self.ofdm = OFDMSignal(n_subcarriers=N, oversample=os_factor)
        self.hpa = RappHPA(A_sat=1.0, p=3.0)
        self.n_symbols = n_symbols
        self.tx_symbols, self.tx_indices = self.ofdm.random_qpsk_symbols(n_symbols)

    # PAPR CCDF for all reducers and all channels
    def run_papr_comparison(self, reducers: dict, n_test=500, channels=["awgn"]):
        all_results = {}
        for ch in channels:
            print(f"\n--- PAPR CCDF for {ch.upper()} channel ---")
            papr_results = {}
            for name, reducer in reducers.items():
                vals = []
                for i in range(n_test):
                    X = self.tx_symbols[i]
                    x_out, meta = reducer.apply(self.ofdm, X)
                    # Apply fading if needed
                    # Channel selection for PAPR CCDF
                    if ch=="awgn":
                        x_out_ch = x_out
                    elif ch=="rayleigh":
                        from channel import rayleigh_fading
                        x_out_ch = rayleigh_fading(x_out)
                    elif ch=="rician":
                        from channel import rician_fading
                        x_out_ch = rician_fading(x_out, K=5)
                    else:
                        raise ValueError(f"Unknown channel: {ch}")
                    vals.append(10*np.log10(np.max(np.abs(x_out_ch)**2)/np.mean(np.abs(x_out_ch)**2)+1e-12))
                papr_results[name] = np.array(vals)
            all_results[ch] = papr_results
            # Plot CCDF per channel
            plot_ccdf(ch, papr_results)
        return all_results

    # BER evaluation with multiple channels
    def evaluate_ber(self, reducers: dict, snr_db_list=[0,5,10,15], n_test=400, si_snr_db=10, channels=["awgn"]):
        results = {}
        for ch in channels:
            print(f"\n--- Evaluating BER over {ch.upper()} channel ---")
            results[ch] = {}
            for name, reducer in reducers.items():
                ser_list=[]; ber_list=[]
                for snr in snr_db_list:
                    sym_err=0; bit_err=0; total_syms=0
                    for i in range(n_test):
                        X = self.tx_symbols[i]
                        x_tx, meta = reducer.apply(self.ofdm, X)

                        # Side information
                        si_ok = True
                        if 'si_idx' in meta and 'U' in meta:
                            idx = meta['si_idx']
                            nbits = int(np.ceil(np.log2(meta['U'])))
                            bits = np.array(list(map(int, np.binary_repr(idx, width=nbits))))
                            decided = bpsk_awgn_tx(bits, si_snr_db)
                            decoded_idx = int(''.join(str(b) for b in decided.tolist()), 2)
                            if decoded_idx != idx:
                                si_ok = False

                        # PA
                        x_pa = self.hpa.apply(x_tx)

                        # Channel
                        if ch=="awgn":
                            y = awgn(x_pa, snr)
                        elif ch=="rayleigh":
                            from channel import rayleigh_fading
                            y = rayleigh_fading(x_pa)
                        elif ch=="rician":
                            from channel import rician_fading
                            y = rician_fading(x_pa, K=5)
                        else:
                            raise ValueError(f"Unknown channel: {ch}")

                        # Receiver
                        X_hat = self.ofdm.ofdm_fft(y)
                        decided_idx = qpsk_decision(X_hat)
                        tx_idx = self.tx_indices[i]

                        # Errors
                        sym_err += np.sum(decided_idx != tx_idx)
                        tx_bits = symbol_indices_to_bits(tx_idx)
                        rx_bits = symbol_indices_to_bits(decided_idx)
                        bit_err += np.sum(tx_bits != rx_bits)
                        total_syms += self.ofdm.N

                    ser_list.append(sym_err/total_syms)
                    ber_list.append(bit_err/(total_syms*2))
                    print(f"{name} ({ch}): SNR={snr}dB -> SER={ser_list[-1]:.3e}, BER={ber_list[-1]:.3e}")

                results[ch][name] = {'snr_db': snr_db_list, 'ser': np.array(ser_list), 'ber': np.array(ber_list)}

            # Plot BER per channel
            plt.figure()
            for name in results[ch]:
                plt.semilogy(results[ch][name]['snr_db'], results[ch][name]['ber'], marker='o', label=name)
            plt.xlabel('SNR (dB)')
            plt.ylabel('BER')
            plt.title(f'BER vs SNR ({ch.upper()} channel)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            filename = f"C:\\OFDM\\PAPR_OFDM\\papr_ofdm_framework\\figures\\BER_SNR_{ch}_{int(time.time())}.pdf"
            plt.savefig(filename, format="pdf", bbox_inches="tight")
            print(f"Saved plot to {filename}")
            plt.show()
        return results

    # PSD plots per reducer and channel
    def psd_all_reducers(self, reducers, channels=["awgn"], n_avg=50, snr=10):
        for ch in channels:
            for name, reducer in reducers.items():
                print(f"\n--- PSD for {name} over {ch.upper()} channel ---")
                signals_before = []; signals_after = []
                for i in range(n_avg):
                    X = self.tx_symbols[i]
                    x_tx, meta = reducer.apply(self.ofdm, X)
                    signals_before.append(x_tx)
                    # Channel
                    if ch=="awgn":
                        y = awgn(self.hpa.apply(x_tx), snr)
                    elif ch=="rayleigh":
                        from channel import rayleigh_fading
                        y = rayleigh_fading(self.hpa.apply(x_tx))
                    elif ch=="rician":
                        from channel import rician_fading
                        y = rician_fading(self.hpa.apply(x_tx), K=5)
                    else:
                        raise ValueError(f"Unknown channel: {ch}")

                    signals_after.append(y)
                xb = np.concatenate(signals_before)
                xa = np.concatenate(signals_after)
                plot_psd_signals(ch, {f'{name}_{ch}_beforeHPA': xb, f'{name}_{ch}_afterHPA': xa})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=str, default="all",
                        choices=["awgn","rayleigh","rician","all"],
                        help="Channel model to evaluate (default: all)")
    args = parser.parse_args()

    exp = Experiment(N=64, os_factor=4, n_symbols=1200)
    reducers = {
        'Original': IdentityReducer(),
        'Clipping': Clipping(clipping_ratio=1.0),
        'Companding': Companding(mu=255),
        'SLM': SLM(n_candidates=8, si_bits=8),
        'PTS': PTS(V=4, phase_set=[1,-1,1j,-1j], search='greedy'),
        'ToneReserveCVX': ToneReservationCVX(reserved_tones=[0,1,62,63])
    }

    print('Running PAPR CCDF comparison...')
    channels_to_run = ["awgn","rayleigh","rician"] if args.channel=="all" else [args.channel]
    papr_results = exp.run_papr_comparison(reducers, n_test=500, channels=channels_to_run)

    print('Evaluating BER (this may take some time)...')
    ber_results = exp.evaluate_ber(reducers, snr_db_list=[0,5,10,15,20], n_test=300, si_snr_db=12, channels=channels_to_run)

    print('PSD before/after HPA per reducer and per channel...')
    exp.psd_all_reducers(reducers, channels=channels_to_run, n_avg=80, snr=12)
