from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from common import scalar_sources, random_mixing, auxiva, save_receipt

torch.manual_seed(1)
np.random.seed(1)
S = scalar_sources(6500)
rng = np.random.default_rng(77)
A_train = random_mixing(rng, 2, cond_max=5.0)
A_test = random_mixing(rng, 2, cond_max=5.0)
X = np.stack([A_train @ S, A_test @ S]).astype(np.complex128)
# Joint IVA is unsupervised. It is allowed to see the unlabeled two-view observations;
# predictor training still uses only the early part of view 0.
Y, _, _ = auxiva(X, n_iter=60)
Y = Y.real

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.r = nn.Linear(in_features, out_features)
        self.i = nn.Linear(in_features, out_features)
    def forward(self, z):
        return torch.complex(self.r(z.real) - self.i(z.imag), self.r(z.imag) + self.i(z.real))

class ResonantPredictor(nn.Module):
    """Small ResonantCortex-style complex predictor: complex mixing + phase-preserving amplitude saturation."""
    def __init__(self, in_features, out_features=2, width=48):
        super().__init__()
        self.ir = nn.Linear(in_features, width)
        self.ii = nn.Linear(in_features, width)
        self.c1 = ComplexLinear(width, width)
        self.c2 = ComplexLinear(width, width)
        self.out = nn.Linear(width * 2, out_features)
    @staticmethod
    def act(z):
        return torch.polar(torch.tanh(torch.abs(z)), torch.angle(z))
    def forward(self, x):
        z = torch.complex(self.ir(x), self.ii(x))
        z = z + self.act(self.c1(z))
        z = z + self.act(self.c2(z))
        return self.out(torch.cat([z.real, z.imag], dim=1))

def windows(Z, lag=12):
    xs, ys = [], []
    for t in range(lag, Z.shape[1] - 1):
        xs.append(Z[:, t-lag:t].reshape(-1))
        ys.append(Z[:, t+1])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)

def cross_geometry_mse(trainZ, testZ):
    xa, ya = windows(trainZ)
    xb, yb = windows(testZ)
    n = len(xa)
    cut = int(0.60 * n)
    test_start = int(0.65 * n)
    sx = StandardScaler().fit(xa[:cut])
    sy = StandardScaler().fit(ya[:cut])
    xtr = sx.transform(xa[:cut]).astype('float32')
    ytr = sy.transform(ya[:cut]).astype('float32')
    xte = sx.transform(xb[test_start:]).astype('float32')
    yte = sy.transform(yb[test_start:]).astype('float32')
    model = ResonantPredictor(xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    lossf = nn.MSELoss()
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr)
    for _ in range(350):
        opt.zero_grad()
        loss = lossf(model(xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(lossf(model(torch.from_numpy(xte)), torch.from_numpy(yte)).item())

raw_mse = cross_geometry_mse(X[0].real, X[1].real)
iva_mse = cross_geometry_mse(Y[0], Y[1])
oracle_mse = cross_geometry_mse(S, S)
receipt = {
    'gate': 'T4 causes buy transfer for a ResonantCortex-style dynamics learner',
    'train_mixing': A_train.tolist(),
    'test_mixing': A_test.tolist(),
    'raw_cross_geometry_mse': raw_mse,
    'iva_cross_geometry_mse': iva_mse,
    'oracle_mse': oracle_mse,
    'iva_over_raw_ratio': iva_mse / raw_mse,
    'pass_rule': 'IVA-coordinate MSE <= 0.20 and <= 0.20 * raw-coordinate MSE',
}
receipt['pass'] = iva_mse <= 0.20 and iva_mse <= 0.20 * raw_mse
print(f"T4 raw={raw_mse:.4f} IVA={iva_mse:.4f} oracle={oracle_mse:.4f} PASS={receipt['pass']}")
save_receipt(Path(__file__).parents[1] / 'results' / 't4.json', receipt)
