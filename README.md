# Tuesday — Matrix Lab

**Status: first runnable gate set, 2026-08-25.**

Tuesday starts where Monday ended: stop asking a matrix to be a metaphor and ask what it can actually recover.

The concrete question is:

> **Can a machine recover enduring dynamical causes from changing mixtures well enough that what it learns about a cause transfers to a new way of observing it?**

The named hidden systems are deliberately familiar from `ResonantCortex`:

- a Lorenz dynamical system;
- a complex quadratic iterator from the Mandelbrot family.

They are mixed, delayed, remixed across views, grouped into dependent subspaces, and finally passed to a small ResonantCortex-style complex predictor. Because Tuesday generated every hidden variable, every separation can be scored against ground truth.

This is not a claim that Lorenz or Mandelbrot dynamics are objects, brain sources, or intelligence. They are **known latent causes** that make a good laboratory.

## The matrix stack

Tuesday keeps several matrices separate:

```text
s(t)                hidden causes
  |
  v
A / H(f)            mixing / propagation
  |
  v
x(t)                observations
  |
  v
W / W(f)            learned demixing transform
  |
  v
y(t)              attempted cause coordinates
  |
  v
predictor            learn dynamics in those coordinates
```

The point is not that everything is a matrix. The point is to know **which matrix answers which question**.

## Development status

These are deterministic development gates with fixed seeds. The final thresholds were frozen for the committed runs **after exploratory implementation**, so treat them as reproducible calibration receipts, not preregistered confirmatory evidence.

| Gate | Question | Registered receipt | Status |
|---|---|---:|---|
| T0 | Can ICA recover two named scalar dynamical causes from a linear mixture where PCA cannot? | ICA `0.9999996`, PCA `0.7071068` mean absolute correlation | PASS |
| T1 | If coordinates inside each cause are dependent, can ICA + dependence grouping recover the two independent **subspaces**? | mean subspace affinity `0.9998791` | PASS |
| T2 | Can IVA recover the same causes across four different mixing matrices with a common source ordering? | mean per-view recovery `0.988304`, common order in 4/4 views | PASS |
| T3 | Does frequency-domain AuxIVA still separate after delayed/convolutive propagation? | mean recovery `0.769793`, max wrong-source leakage `0.017872` | PASS, modest |
| T4 | Does separation buy something downstream? Can one ResonantCortex-style predictor transfer to a new sensor geometry? | raw MSE `2.76416`; IVA-coordinate MSE `0.08215`; oracle `0.00116` | PASS — strongest result |
| T5 | Where does the linear matrix story break? | linear ICA `0.9999996`; after `tanh(1.2 A s)`, `0.6697353` | EXPECTED BOUNDARY CONFIRMED |
| T6 | Show only the working mechanisms and their limits in a browser | [`index.html`](index.html) | BUILT |

## T0 — algorithm cocktail party

`experiments/gate_t0_algorithm_cocktail.py`

Two scalar source signals are hidden by

```text
x(t) = A s(t)
```

with a deliberately awkward source rotation. PCA lands on a 45-degree mixture of the sources; FastICA uses non-Gaussian independence and recovers them almost exactly.

This gate earns only the statement that **blind linear source separation works in its intended model class**.

## T1 — systems, not coordinates

`experiments/gate_t1_independent_subspaces.py`

Now the hidden state is five-dimensional:

```text
Lorenz        = [x, y, z]        dependent internally
Quadratic     = [Re z, Im z]     dependent internally

Lorenz  independent of  Quadratic
```

A random 5x5 matrix destroys the visible grouping.

Ordinary ICA is used as preprocessing, then its components are grouped by estimated mutual dependence. This follows the **ISA separation principle**: ICA first, then cluster ICA elements into statistically dependent subspaces.

The recovered 3-D and 2-D spans match the true hidden subspaces with mean affinity `0.9998791`.

The important ambiguity is preserved: **ISA should recover the subspace, not the privileged axes inside it.**

## T2 — many eyes, same causes

`experiments/gate_t2_many_eyes_iva.py`

Four observers see the same two underlying signals through four different 2x2 matrices:

```text
x[k](t) = A[k] s(t)
```

Separate FastICA runs recover the signals extremely well but one run flips their ordering. That is not a bug; ICA has a permutation ambiguity independently in each dataset.

The compact AuxIVA implementation treats the four views as linked datasets. It recovers the signals at about `0.9883` mean absolute correlation in every view and keeps one common ordering across all four.

This is Tuesday's cleanest demonstration of what IVA buys:

> **different local coordinates, one linked source identity.**

## T3 — propagation, not instantaneous mixing

`experiments/gate_t3_convolutive_auxiva.py`

AuxIVA was developed for the harder case where propagation introduces delays/reverberation. To use the algorithm in a model class it actually fits, Lorenz and quadratic dynamics modulate independent super-Gaussian broadband carriers; the two carriers are then mixed through delayed FIR paths.

The committed run gets:

```text
mean |corr| with true sources       0.769793
max correlation with wrong source   0.017872
```

The raw sensors already correlate fairly strongly with one source each (`0.755043` mean), so this is **not** a dramatic headline win. The cleaner result is suppression of cross-source leakage.

## T4 — the reason Tuesday exists

`experiments/gate_t4_resonant_transfer.py`

This gate gives source separation a job.

Two sensor worlds observe the same hidden dynamics through different matrices. A small complex-valued predictor derived from the ResonantCortex idea is trained on early data from world A and tested on later data expressed in world B coordinates.

```text
RAW SENSOR COORDINATES
train on A1, test on A2          MSE 2.76416

IVA-LINKED CAUSE COORDINATES
train on view 1, test on view 2  MSE 0.08215

ORACLE TRUE SOURCES               MSE 0.00116
```

So on this controlled synthetic problem, the learned dynamics transfer about **34x better** after IVA has put the two observations into linked source coordinates.

That is a much more useful claim than “ICA finds objects”:

> **source coordinates can make a learned dynamical law substantially more invariant to the observation geometry.**

Important limitation: joint IVA is fitted unsupervised on the two-view observations before the forecasting score is computed. This is a multi-view/transductive representation experiment, not yet a strict online “new sensor arrives unseen” test. A later gate should freeze or adapt the demixer under a declared online protocol, where Dynamic IVA / IVE becomes relevant.

## T5 — deliberately break the assumption

`experiments/gate_t5_break_linear_matrix.py`

Keep the same source signals and linear matrix, but change the sensor to

```text
x = tanh(1.2 A s)
```

FastICA falls from essentially perfect recovery to `0.6697` mean absolute correlation.

That is a **success of the gate because the model failed where its assumptions were violated**.

Do not tune linear ICA until this passes. The scientific next step is identifiable nonlinear ICA, which requires additional structure such as temporal or auxiliary information.

## Where IVE fits

Tuesday currently implements full ICA / ISA-style grouping / IVA / AuxIVA. It does **not** pretend that “select one row after full IVA” is an IVE algorithm.

Independent Vector Extraction is the natural next economy gate:

```text
many latent causes
      |
      v
observations
      |
      +---- full IVA: explain all sources
      |
      +---- IVE: recover only the source vector of interest
```

The useful test will be computational and predictive: if only one source matters downstream, can a proper IVE implementation recover it with less work or adapt it more robustly than separating everything?

## T6 browser demo

Open `index.html`, or use GitHub Pages when enabled:

```text
https://anttiluode.github.io/Tuesday/
```

The page contains:

- a live 2x2 mixing-matrix toy with Lorenz + complex-quadratic signals;
- the committed gate receipts;
- a visual explanation of ICA vs ISA vs IVA vs AuxIVA;
- the T4 transfer result;
- the T5 nonlinear boundary;
- links to the relevant papers and predecessor repos.

The browser's **UNMIX** button intentionally uses the known matrix inverse so the geometry is visible. It is not presented as ICA. The blind algorithms live in the Python gates.

## Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt

python experiments/gate_t0_algorithm_cocktail.py
python experiments/gate_t1_independent_subspaces.py
python experiments/gate_t2_many_eyes_iva.py
python experiments/gate_t3_convolutive_auxiva.py
python experiments/gate_t4_resonant_transfer.py
python experiments/gate_t5_break_linear_matrix.py
```

Receipts are written to `results/`.

## Lineage

- [BrainArchitectureAnalyzer2](https://github.com/anttiluode/BrainArchitectureAnalyzer2) — made the distinction between observation dependence, demixing coordinates and transition matrices explicit.
- [ResonantCortex](https://github.com/anttiluode/ResonantCortex) — supplied the Lorenz/Mandelbrot dynamical learners and the complex-valued predictor idea.
- [Monday](https://github.com/anttiluode/Monday) — source separation became the mature language for “mixtures -> causes”; compression added the rule that a factorization must buy prediction, bits, search or distortion.

## Literature map

- A. Hyvärinen, E. Oja. **Independent component analysis: algorithms and applications.** Neural Networks 13 (2000). DOI: https://doi.org/10.1016/S0893-6080(00)00026-5
- T. Kim, T. Eltoft, T.-W. Lee. **Independent Vector Analysis: An Extension of ICA to Multivariate Components.** ICA 2006. DOI: https://doi.org/10.1007/11679363_21
- Z. Szabó, B. Póczos, A. Lőrincz. **Separation theorem for independent subspace analysis and its consequences.** Pattern Recognition 45 (2012). DOI: https://doi.org/10.1016/j.patcog.2011.09.007
- N. Ono. **Stable and fast update rules for independent vector analysis based on auxiliary function technique.** WASPAA 2011. DOI: https://doi.org/10.1109/ASPAA.2011.6082320
- M. Arvila, K. Nordhausen, M. Sipilä, S. Taskinen. **Independent vector analysis — an introduction for statisticians.** arXiv: https://arxiv.org/abs/2506.16175
- Z. Koldovský et al. **Dynamic Independent Component/Vector Analysis: Time-Variant Linear Mixtures Separable by Time-Invariant Beamformers.** IEEE TSP 69 (2021). https://doi.org/10.1109/TSP.2021.3068626
- A. Hyvärinen, H. Sasaki, R. E. Turner. **Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning.** AISTATS 2019. https://proceedings.mlr.press/v89/hyvarinen19a.html
- B. Bozkurt et al. **Normative Networks for Source Separation via Local Plasticity and Dendritic Computation.** arXiv: https://arxiv.org/abs/2605.19965
- R. Guo, Z. Luo, M. Li. **A Survey of Optimization Methods for Independent Vector Analysis in Audio Source Separation.** Sensors 23, 493 (2023). https://doi.org/10.3390/s23010493

## Current claim boundary

Tuesday has **not** discovered object perception or intelligence.

It has produced a clean synthetic result worth carrying forward:

> **When two observation systems contain the same hidden dynamical causes in different linear coordinates, IVA can align those causes across views, and a small dynamics learner can transfer far better in the aligned cause coordinates than in the raw sensor coordinates.**

The next serious step is to make that survive harder source processes, changing mixtures, partial overlap, noise, and then a real measurement problem.
