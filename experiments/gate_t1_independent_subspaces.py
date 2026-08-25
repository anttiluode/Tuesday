from pathlib import Path
import itertools
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.feature_selection import mutual_info_regression
from common import lorenz, quadratic_orbit, random_mixing, save_receipt

# One independent object is 3-D Lorenz; another is a 2-D complex quadratic orbit.
# Coordinates *within* each object are dependent. The scientifically appropriate target
# is therefore two independent subspaces, not five scalar independent sources.
n = 6000
S = np.vstack([lorenz(n=n), quadratic_orbit(n=n)])
rng = np.random.default_rng(11)
A = random_mixing(rng, 5, cond_max=5.0)
X = A @ S

# ISA separation principle: ICA preprocessing, then group the ICA elements by dependence.
ica = FastICA(n_components=5, whiten='unit-variance', random_state=5,
              max_iter=10000, tol=1e-5, fun='cube')
Y = ica.fit_transform(X.T).T

idx = np.linspace(0, n - 1, 2000, dtype=int)
MI = np.zeros((5, 5))
for i in range(5):
    for j in range(i + 1, 5):
        a = mutual_info_regression(Y[i, idx, None], Y[j, idx], random_state=0, n_neighbors=10)[0]
        b = mutual_info_regression(Y[j, idx, None], Y[i, idx], random_state=0, n_neighbors=10)[0]
        MI[i, j] = MI[j, i] = 0.5 * (a + b)

def partition_objective(g1):
    g1 = set(g1)
    g2 = set(range(5)) - g1
    within = sum(MI[i, j] for g in (g1, g2) for i in g for j in g if i < j)
    cross = sum(MI[i, j] for i in g1 for j in g2)
    return float(within - cross), sorted(g1), sorted(g2)

parts = [partition_objective(g) for g in itertools.combinations(range(5), 3)]
parts.sort(reverse=True)
objective, g3, g2 = parts[0]

def subspace_score(Arows, Brows):
    Qa, _ = np.linalg.qr(Arows.T)
    Qb, _ = np.linalg.qr(Brows.T)
    d = min(Qa.shape[1], Qb.shape[1])
    return float(np.linalg.norm(Qa.T @ Qb, 'fro') ** 2 / d)

direct = 0.5 * (subspace_score(Y[g3], S[:3]) + subspace_score(Y[g2], S[3:]))
swapped = 0.5 * (subspace_score(Y[g3], S[3:]) + subspace_score(Y[g2], S[:3]))
best = max(direct, swapped)

receipt = {
    'gate': 'T1 independent systems, not independent coordinates',
    'mixing_matrix': A.tolist(),
    'ica_iterations': int(ica.n_iter_),
    'pairwise_mutual_information': MI.tolist(),
    'chosen_groups': [g3, g2],
    'partition_objective': objective,
    'direct_subspace_score': direct,
    'swapped_subspace_score': swapped,
    'best_subspace_score': best,
    'pass_rule': 'best mean subspace affinity >= 0.95',
}
receipt['pass'] = best >= 0.95
print(f"T1 groups={g3}|{g2} subspace={best:.4f} PASS={receipt['pass']}")
save_receipt(Path(__file__).parents[1] / 'results' / 't1.json', receipt)
