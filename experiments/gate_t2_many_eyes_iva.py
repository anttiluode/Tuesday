from pathlib import Path
import numpy as np
from sklearn.decomposition import FastICA
from common import scalar_sources, random_mixing, auxiva, match_sources, save_receipt

S = scalar_sources(5000)
rng = np.random.default_rng(23)
K = 4
As, Xs = [], []
for _ in range(K):
    A = random_mixing(rng, 2, cond_max=5.0)
    As.append(A)
    Xs.append(A @ S)
X = np.stack(Xs).astype(np.complex128)  # dataset/view x sensor x time

Y, _, _ = auxiva(X, n_iter=60)
iva_scores, iva_orders = [], []
for k in range(K):
    r = match_sources(Y[k].real, S)
    iva_scores.append(r['mean_abs_corr'])
    iva_orders.append([j for i, j, _ in sorted(r['assignment'])])

# Attacker: solve each view with an independent ICA. Separation is excellent, but
# source order is free to permute separately in each dataset.
ica_scores, ica_orders = [], []
for k in range(K):
    y = FastICA(n_components=2, whiten='unit-variance', random_state=100 + k,
                max_iter=3000, tol=1e-6).fit_transform(X[k].real.T).T
    r = match_sources(y, S)
    ica_scores.append(r['mean_abs_corr'])
    ica_orders.append([j for i, j, _ in sorted(r['assignment'])])

iva_aligned = all(order == iva_orders[0] for order in iva_orders[1:])
ica_permutation_disagreement = any(order != ica_orders[0] for order in ica_orders[1:])
receipt = {
    'gate': 'T2 many eyes, same causes',
    'n_views': K,
    'mixing_matrices': [a.tolist() for a in As],
    'iva_mean_corr_per_view': iva_scores,
    'iva_orders': iva_orders,
    'separate_ica_mean_corr_per_view': ica_scores,
    'separate_ica_orders': ica_orders,
    'iva_aligned': iva_aligned,
    'separate_ica_permutation_disagreement': ica_permutation_disagreement,
    'pass_rule': 'IVA mean recovery >= 0.95 in every view and one common source order across views',
}
receipt['pass'] = min(iva_scores) >= 0.95 and iva_aligned
print(f"T2 IVA mean={np.mean(iva_scores):.4f} aligned={iva_aligned} separate-ICA-orders={ica_orders} PASS={receipt['pass']}")
save_receipt(Path(__file__).parents[1] / 'results' / 't2.json', receipt)
