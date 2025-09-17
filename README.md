PAPR OFDM Framework (modular)

Files:
- ofdm.py           : OFDM generation and FFT/IFFT helpers
- reducers.py       : PAPR reduction classes (Clipping, Companding, SLM, PTS, ToneReservationCVX)
- channel.py        : HPA model (Rapp) and AWGN / BPSK helpers for SI channel
- analysis.py       : CCDF and PSD plotting helpers
- main.py           : Example experiments (PAPR CCDF, BER evaluation with SI, PSD before/after HPA)

Usage:
1. install requirements: pip install -r requirements.txt
2. run: python main.py
=======
# papr_ofdm_framework
study the effects of PAPR on OFDM signal in transmission over differents channel