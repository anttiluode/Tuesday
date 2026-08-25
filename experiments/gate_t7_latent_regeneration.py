from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(2)

SIZE = 20
TRAIN_T = 350
TEST_T = 150
LATENT_COMPLEX = 2
EPOCHS = 600
SEEDS = (0, 1, 2)
MASK = (6, 14, 6, 14)
LAGS = (1, 2, 4, 8)


def make_world(total: int = TRAIN_T + TEST_T) -> np.ndarray:
    """Two independently evolving visible objects with known oscillator causes."""
    t = np.arange(total, dtype=np.float64)
    phi1 = 0.075 * t + 0.15 * np.sin(0.013 * t)
    phi2 = 0.117 * t + 0.12 * np.sin(0.021 * t + 0.7)

    p1 = np.stack([0.55 * np.cos(phi1) - 0.15, 0.45 * np.sin(phi1)], axis=1)
    p2 = np.stack([0.48 * np.cos(phi2) + 0.18, 0.52 * np.sin(phi2 + 0.4)], axis=1)

    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    gx = (xx / (SIZE - 1)) * 2.0 - 1.0
    gy = (yy / (SIZE - 1)) * 2.0 - 1.0

    frames = []
    for a, b in zip(p1, p2):
        g1 = np.exp(-((gx - a[0]) ** 2 + (gy - a[1]) ** 2) / (2 * 0.12**2))
        g2 = 0.85 * np.exp(-((gx - b[0]) ** 2 + (gy - b[1]) ** 2) / (2 * 0.10**2))
        frames.append(np.clip(g1 + g2, 0.0, 1.0).astype(np.float32))
    return np.stack(frames)


class TinyCausalAutoencoder(nn.Module):
    """Small visual autoencoder with ResonantCortex-style diagonal complex dynamics."""

    def __init__(self) -> None:
        super().__init__()
        npx = SIZE * SIZE
        self.encoder = nn.Sequential(
            nn.Linear(npx, 96),
            nn.Tanh(),
            nn.Linear(96, 48),
            nn.Tanh(),
            nn.Linear(48, 2 * LATENT_COMPLEX),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * LATENT_COMPLEX, 48),
            nn.Tanh(),
            nn.Linear(48, 96),
            nn.Tanh(),
            nn.Linear(96, npx),
            nn.Sigmoid(),
        )
        self.log_rho = nn.Parameter(torch.zeros(LATENT_COMPLEX))
        self.omega = nn.Parameter(torch.randn(LATENT_COMPLEX) * 0.05)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        v = self.encoder(x)
        return torch.complex(v[:, :LATENT_COMPLEX], v[:, LATENT_COMPLEX:])

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        v = torch.cat([z.real, z.imag], dim=1)
        return self.decoder(v)

    def step(self, z: torch.Tensor) -> torch.Tensor:
        rho = torch.sigmoid(self.log_rho) * 0.15 + 0.90
        multiplier = torch.polar(rho, self.omega)
        return z * multiplier


def _normalize_complex(z: torch.Tensor) -> torch.Tensor:
    q = z - z.mean(dim=0, keepdim=True)
    scale = torch.sqrt((q.abs() ** 2).mean(dim=0, keepdim=True) + 1e-6)
    return q / scale


def instant_decorrelation(z: torch.Tensor) -> torch.Tensor:
    """Lag-0 attacker: remove cross-component second-order dependence only."""
    q = _normalize_complex(z)
    cov = (q.T @ q.conj()) / q.shape[0]
    off = cov - torch.diag(torch.diagonal(cov))
    return (off.abs() ** 2).mean()


def lagged_decorrelation(z: torch.Tensor) -> torch.Tensor:
    """SOBI-inspired loss: suppress cross-component covariance at several lags."""
    q = _normalize_complex(z)
    penalties = []
    for lag in (0, *LAGS):
        if lag == 0:
            a, b = q, q
        else:
            a, b = q[lag:], q[:-lag]
        cov = (a.T @ b.conj()) / a.shape[0]
        off = cov - torch.diag(torch.diagonal(cov))
        penalties.append((off.abs() ** 2).mean())
    return torch.stack(penalties).mean()


def train(kind: str, frames: np.ndarray, seed: int) -> TinyCausalAutoencoder:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TinyCausalAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    x = torch.tensor(frames[:TRAIN_T].reshape(TRAIN_T, -1))

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        z = model.encode(x)
        recon = model.decode(z)
        loss = ((recon - x) ** 2).mean()

        if kind in {"B", "C0", "C"}:
            predicted = model.decode(model.step(z[:-1]))
            loss = loss + 1.5 * ((predicted - x[1:]) ** 2).mean()

        if kind == "C0":
            loss = loss + 0.08 * instant_decorrelation(z)
        elif kind == "C":
            loss = loss + 0.08 * lagged_decorrelation(z)

        loss.backward()
        optimizer.step()

    return model


def mask_indices() -> tuple[torch.Tensor, torch.Tensor]:
    y0, y1, x0, x1 = MASK
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[y0:y1, x0:x1] = True
    missing = torch.tensor(mask.reshape(-1))
    return missing, ~missing


def latent_completion(
    model: TinyCausalAutoencoder,
    kind: str,
    previous_clean: torch.Tensor,
    current_masked: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    """Fit visible pixels while a learned dynamics prior regenerates the hidden patch."""
    with torch.no_grad():
        if kind == "A":
            z0 = model.encode(current_masked)
            prior = None
        else:
            prior = model.step(model.encode(previous_clean))
            z0 = prior.clone()

    v = torch.cat([z0.real, z0.imag], dim=1).detach().clone().requires_grad_(True)
    optimizer = optim.Adam([v], lr=0.08)

    for _ in range(12):
        optimizer.zero_grad()
        z = torch.complex(v[:, :LATENT_COMPLEX], v[:, LATENT_COMPLEX:])
        reconstruction = model.decode(z)
        loss = ((reconstruction[:, visible] - current_masked[:, visible]) ** 2).mean()
        if prior is not None:
            p = torch.cat([prior.real, prior.imag], dim=1)
            loss = loss + 0.4 * ((v - p) ** 2).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        z = torch.complex(v[:, :LATENT_COMPLEX], v[:, LATENT_COMPLEX:])
        return model.decode(z)


def evaluate(model: TinyCausalAutoencoder, kind: str, frames: np.ndarray) -> dict[str, object]:
    x = torch.tensor(frames[TRAIN_T:].reshape(TEST_T, -1))
    missing, visible = mask_indices()

    with torch.no_grad():
        z = model.encode(x)
        clean_recon = model.decode(z)
        reconstruction_mse = float(((clean_recon - x) ** 2).mean())
        if kind == "A":
            one_step_mse = None
        else:
            pred = model.decode(model.step(z[:-1]))
            one_step_mse = float(((pred - x[1:]) ** 2).mean())

    patch_errors = []
    for t in range(1, 60):
        # Ignore trivial examples where the hidden patch contains almost no signal.
        if float(x[t, missing].mean()) <= 0.015:
            continue
        masked = x[t:t + 1].clone()
        masked[:, missing] = 0.0
        completed = latent_completion(model, kind, x[t - 1:t], masked, visible)
        patch_errors.append(float(((completed[:, missing] - x[t:t + 1, missing]) ** 2).mean()))

    return {
        "clean_reconstruction_mse": reconstruction_mse,
        "one_step_prediction_mse": one_step_mse,
        "missing_patch_mse": float(np.mean(patch_errors)),
        "n_nontrivial_masked_frames": len(patch_errors),
    }


def main() -> None:
    frames = make_world()
    per_seed: dict[str, dict[str, dict[str, object]]] = {}
    kinds = ("A", "B", "C0", "C")

    for seed in SEEDS:
        seed_results = {}
        for kind in kinds:
            model = train(kind, frames, seed)
            seed_results[kind] = evaluate(model, kind, frames)
        per_seed[str(seed)] = seed_results

    mean_patch = {
        kind: float(np.mean([per_seed[str(seed)][kind]["missing_patch_mse"] for seed in SEEDS]))
        for kind in kinds
    }
    mean_recon = {
        kind: float(np.mean([per_seed[str(seed)][kind]["clean_reconstruction_mse"] for seed in SEEDS]))
        for kind in kinds
    }

    receipt = {
        "status": "exploratory",
        "question": (
            "Does a structured predictive latent make a decoder regenerate a hidden image patch "
            "better than reconstruction or prediction alone?"
        ),
        "models": {
            "A": "reconstruction only",
            "B": "reconstruction + diagonal complex prediction",
            "C0": "B + lag-0 complex decorrelation attacker",
            "C": "B + SOBI-inspired cross-component decorrelation at lags 0,1,2,4,8",
        },
        "setup": {
            "image_size": SIZE,
            "train_frames": TRAIN_T,
            "test_frames": TEST_T,
            "latent": f"{LATENT_COMPLEX} complex units ({2 * LATENT_COMPLEX} real degrees)",
            "mask": MASK,
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
        },
        "per_seed": per_seed,
        "mean_clean_reconstruction_mse": mean_recon,
        "mean_missing_patch_mse": mean_patch,
        "relative_patch_mse_C_vs_A": mean_patch["C"] / mean_patch["A"],
        "relative_patch_mse_C_vs_B": mean_patch["C"] / mean_patch["B"],
        "relative_patch_mse_C_vs_C0": mean_patch["C"] / mean_patch["C0"],
        "interpretation": (
            "If C beats A/B, structured latent coordinates bought regeneration. "
            "C0 is the attribution attacker: if C does not beat C0, the current result cannot "
            "be credited specifically to lagged/SOBI structure rather than ordinary decorrelation."
        ),
    }

    Path("results").mkdir(exist_ok=True)
    Path("results/t7_latent_regeneration.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
