from pathlib import Path
import numpy as np
from scipy import signal
from common import lorenz, quadratic_orbit, zscore_rows, stft_multi, auxiva, istft_sources, corr_matrix, match_sources, save_receipt

# AuxIVA is designed for convolutive mixtures and super-Gaussian broadband signals.
# We therefore let the named dynamical systems modulate independent broadband carriers.
rng = np.random.default_rng(42)
n = 12000
L = lorenz(n=n)[0]
Q = quadratic_orbit(n=n)[0]
env1 = 0.2 + 1.5 / (1.0 + np.exp(-L))
env2 = 0.2 + 1.5 / (1.0 + np.exp(-Q))
b, a = signal.butter(4, 0.35)
u1 = signal.lfilter(b, a, rng.laplace(size=n))
u2 = signal.lfilter(b, a, rng.laplace(size=n))
S = zscore_rows(np.stack([env1 * u1, env2 * u2]))

def delay(x, d):
    return np.concatenate([np.zeros(d), x[:-d]]) if d else x.copy()

X = np.stack([
    S[0] + 0.90 * delay(S[1], 7) + 0.20 * delay(S[1], 17),
    -0.80 * delay(S[0], 4) + S[1] - 0.15 * delay(S[0], 13),
])

_, _, Z = stft_multi(X, fs=200.0, nperseg=256, noverlap=192)
Yf, _, _ = auxiva(Z, n_iter=80)
Y = istft_sources(Yf, fs=200.0, nperseg=256, noverlap=192, length=n)
score = match_sources(Y, S)
C = np.abs(corr_matrix(Y, S))
assign = {i: j for i, j, _ in score['assignment']}
leak = max(C[i, 1 - assign[i]] for i in range(2))
raw = match_sources(X, S)

receipt = {
    'gate': 'T3 convolutive propagation / AuxIVA',
    'auxiva': score,
    'raw_sensor_match': raw,
    'max_wrong_source_leakage': float(leak),
    'pass_rule': 'AuxIVA mean |corr| >= 0.72 and max wrong-source leakage <= 0.05',
}
receipt['pass'] = score['mean_abs_corr'] >= 0.72 and leak <= 0.05
print(f"T3 AuxIVA={score['mean_abs_corr']:.4f} leakage={leak:.4f} PASS={receipt['pass']}")
save_receipt(Path(__file__).parents[1] / 'results' / 't3.json', receipt)
