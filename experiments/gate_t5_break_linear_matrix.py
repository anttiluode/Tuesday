from pathlib import Path
import numpy as np
from sklearn.decomposition import FastICA
from common import scalar_sources, match_sources, save_receipt

S = scalar_sources(7000)
theta = np.pi / 4.0
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
A = np.diag([2.0, 0.5]) @ R.T
U = A @ S

# Linear control.
Ylin = FastICA(n_components=2, whiten='unit-variance', random_state=3,
               max_iter=5000, tol=1e-6).fit_transform(U.T).T
linear = match_sources(Ylin, S)

# Deliberately violate x = A s with a smooth nonlinear sensor transform.
Xnon = np.tanh(1.2 * U)
Ynon = FastICA(n_components=2, whiten='unit-variance', random_state=3,
               max_iter=5000, tol=1e-6).fit_transform(Xnon.T).T
nonlinear = match_sources(Ynon, S)

receipt = {
    'gate': 'T5 break the linear matrix',
    'linear_control': linear,
    'nonlinear_sensor_mix': 'x = tanh(1.2 A s)',
    'linear_ica_on_nonlinear_mix': nonlinear,
    'pass_rule': 'boundary confirmed if linear control >= 0.95 but nonlinear recovery <= 0.75',
    'interpretation': 'PASS here means the linear ICA model failed where it should; use identifiable nonlinear-ICA assumptions rather than tuning linear ICA.',
}
receipt['pass'] = linear['mean_abs_corr'] >= 0.95 and nonlinear['mean_abs_corr'] <= 0.75
print(f"T5 linear={linear['mean_abs_corr']:.4f} nonlinear={nonlinear['mean_abs_corr']:.4f} boundary-PASS={receipt['pass']}")
save_receipt(Path(__file__).parents[1] / 'results' / 't5.json', receipt)
