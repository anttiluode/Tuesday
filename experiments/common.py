from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy import signal
from scipy.optimize import linear_sum_assignment

EPS = 1e-10


def zscore_rows(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, float)
    return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + EPS)


def lorenz(n=6000, dt=0.01, burn=1000, state=(1.0, 1.0, 1.0)) -> np.ndarray:
    total = n + burn
    t = np.arange(total) * dt
    def f(_t, y):
        x, yy, z = y
        return [10.0 * (yy - x), x * (28.0 - z) - yy, x * yy - (8.0 / 3.0) * z]
    sol = solve_ivp(f, (0.0, t[-1]), state, t_eval=t, rtol=1e-9, atol=1e-11)
    return zscore_rows(sol.y[:, burn:])


def quadratic_orbit(n=6000, burn=1000, c=-0.745 + 0.113j, z0=0j) -> np.ndarray:
    # Bounded/interesting complex quadratic orbit, used as a Mandelbrot-family dynamical source.
    z = complex(z0)
    vals = []
    for i in range(n + burn):
        z = z*z + c
        if not np.isfinite(z.real) or not np.isfinite(z.imag) or abs(z) > 4:
            z = 0j
        if i >= burn:
            vals.append(z)
    arr = np.asarray(vals, np.complex128)
    return zscore_rows(np.stack([arr.real, arr.imag]))


def scalar_sources(n=6000) -> np.ndarray:
    L = lorenz(n=n)
    Q = quadratic_orbit(n=n)
    s1 = L[0]
    s2 = 0.75 * Q[0] + 0.35 * Q[1]
    return zscore_rows(np.stack([s1, s2]))


def random_mixing(rng: np.random.Generator, n: int, cond_max=6.0) -> np.ndarray:
    for _ in range(1000):
        A = rng.normal(size=(n, n))
        c = np.linalg.cond(A)
        if np.isfinite(c) and c < cond_max:
            return A
    raise RuntimeError('could not draw well-conditioned mixing matrix')


def corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = zscore_rows(A)
    B = zscore_rows(B)
    return np.corrcoef(A, B)[:A.shape[0], A.shape[0]:]


def match_sources(est: np.ndarray, truth: np.ndarray) -> dict:
    C = np.abs(corr_matrix(est, truth))
    r, c = linear_sum_assignment(-C)
    vals = C[r, c]
    return {
        'mean_abs_corr': float(np.mean(vals)),
        'min_abs_corr': float(np.min(vals)),
        'assignment': [(int(i), int(j), float(v)) for i, j, v in zip(r, c, vals)],
        'corr': C.tolist(),
    }


def whiten(X: np.ndarray):
    X = X - X.mean(axis=1, keepdims=True)
    C = X @ X.T / X.shape[1]
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, EPS)
    V = (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T
    return V @ X, V


def whiten_frequency_bins(X: np.ndarray):
    F, C, T = X.shape
    Xw = np.empty_like(X, dtype=np.complex128)
    V = np.empty((F, C, C), dtype=np.complex128)
    for f in range(F):
        cov = (X[f] @ X[f].conj().T) / max(1, T)
        cov += EPS * np.eye(C)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals.real, EPS)
        vf = (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.conj().T
        V[f] = vf
        Xw[f] = vf @ X[f]
    return Xw, V


def auxiva(X: np.ndarray, n_iter: int = 40):
    """Compact determined complex AuxIVA-IP, shared with the Monday/BAA2 line."""
    Xw, Vwhite = whiten_frequency_bins(np.asarray(X, np.complex128))
    F, C, T = Xw.shape
    W = np.tile(np.eye(C, dtype=np.complex128), (F, 1, 1))
    eye = np.eye(C, dtype=np.complex128)
    for _ in range(int(n_iter)):
        Y = np.einsum('fsc,fct->fst', W, Xw, optimize=True)
        r = np.sqrt(np.sum(np.abs(Y) ** 2, axis=0) + EPS)
        phi = 1.0 / np.maximum(r, 1e-7)
        for s in range(C):
            weights = phi[s][None, :].repeat(F, axis=0)
            covs = np.einsum('fct,ft,fdt->fcd', Xw, weights, Xw.conj(), optimize=True) / max(1, T)
            covs += EPS * eye[None, :, :]
            for f in range(F):
                cov = covs[f]
                M = W[f] @ cov
                try:
                    w = np.linalg.solve(M, eye[:, s])
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(M, eye[:, s], rcond=None)[0]
                denom = np.sqrt(np.real(w.conj().T @ cov @ w) + EPS)
                w /= denom
                W[f, s, :] = w.conj()
    Y = np.einsum('fsc,fct->fst', W, Xw, optimize=True)
    return Y, W, Vwhite


def stft_multi(X: np.ndarray, fs=100.0, nperseg=256, noverlap=192):
    Z = []
    for row in X:
        f, t, z = signal.stft(row, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
                              boundary='zeros', padded=True)
        Z.append(z)
    return f, t, np.transpose(np.stack(Z), (1, 0, 2))


def istft_sources(Y: np.ndarray, fs=100.0, nperseg=256, noverlap=192, length=None):
    F, S, T = Y.shape
    outs = []
    for s in range(S):
        _, y = signal.istft(Y[:, s, :], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
                            input_onesided=True, boundary=True)
        if length is not None:
            y = y[:length]
        outs.append(y)
    m = min(map(len, outs))
    return np.stack([o[:m] for o in outs])


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'Object of type {type(o).__name__} is not JSON serializable')


def save_receipt(path: str | Path, data: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding='utf-8')
