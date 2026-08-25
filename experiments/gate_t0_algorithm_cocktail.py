from pathlib import Path
import numpy as np
from sklearn.decomposition import FastICA, PCA
from common import scalar_sources, match_sources, save_receipt

# Predeclared geometry: equal-variance sources are mixed by a 45-degree source rotation,
# then stretched in sensor space. PCA can whiten/rotate the cloud but should not recover
# source identity; ICA gets the non-Gaussian independence cue.
S = scalar_sources(7000)
theta = np.pi / 4.0
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
A = np.diag([2.0, 0.5]) @ R.T
X = A @ S

ica = FastICA(n_components=2, whiten='unit-variance', random_state=3, max_iter=3000, tol=1e-6)
Y_ica = ica.fit_transform(X.T).T
pca = PCA(n_components=2, random_state=3)
Y_pca = pca.fit_transform(X.T).T

r_ica = match_sources(Y_ica, S)
r_pca = match_sources(Y_pca, S)
receipt = {
    'gate': 'T0 algorithm cocktail party',
    'sources': ['Lorenz x(t)', 'complex quadratic / Mandelbrot-family scalar orbit'],
    'mixing_matrix': A.tolist(),
    'ica': r_ica,
    'pca': r_pca,
    'pass_rule': 'ICA mean |corr| >= 0.95 and PCA mean |corr| <= 0.80',
}
receipt['pass'] = r_ica['mean_abs_corr'] >= 0.95 and r_pca['mean_abs_corr'] <= 0.80
print(f"T0 ICA={r_ica['mean_abs_corr']:.4f} PCA={r_pca['mean_abs_corr']:.4f} PASS={receipt['pass']}")
save_receipt(Path(__file__).parents[1] / 'results' / 't0.json', receipt)
