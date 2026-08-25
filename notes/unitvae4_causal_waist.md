# UnitVAE4 -> causal waist -> Resonant dynamics

`unitvae4` already contains the useful skeleton:

```text
frame
  -> adaptive encoder
  -> latent
  -> adaptive decoder
  -> reconstructed frame
```

The new question is not merely whether the latent reconstructs. It is whether a **small temporal waist** can become a coordinate system for persistent dynamical causes.

## Proposed architecture

```text
frame x(t)
    |
    v
visual encoder
    |
    v
small complex waist q(t)        8-64 real DOF, not the full 4x64x64 teacher latent
    |
    +-----------------------------+
    |                             |
    v                             v
separation / identifiability      resonant dynamics
objective                         q(t) -> q(t+1)
    |                             |
    +--------------+--------------+
                   v
                decoder
                   |
                   v
             frame x_hat(t)
```

The VAE/student has one job: preserve enough visible information to render.

The separation layer has another: make the waist coordinates simple under a declared source model.

The resonant layer has a third: learn how those coordinates move through time.

## First A/B/C test

`experiments/gate_t7_latent_regeneration.py` is deliberately much smaller than UnitVAE4. It asks whether the idea has any measurable bite before attaching Stable Video Diffusion or a webcam.

- **A** — reconstruction only.
- **B** — reconstruction + a diagonal complex one-step dynamics model.
- **C** — B + a SOBI-inspired penalty that suppresses cross-component covariance at lags 0, 1, 2, 4, 8.
- **C0 attacker** — B + lag-0 decorrelation only.

The hidden world contains two independently evolving visible objects. A central image patch is removed at test time. The latent is adjusted to match the visible pixels; predictive models also receive a one-step latent prior from the previous clean frame. The score is MSE **inside the missing patch**.

The first exploratory receipt averaged over three seeds is:

```text
A   reconstruction only                     0.03210
B   reconstruction + prediction             0.03210
C0  B + instantaneous decorrelation         0.01927
C   B + lagged decorrelation                 0.01908
```

So a structured bottleneck cuts missing-patch MSE to about **59% of A/B** in this toy.

But C is only about 1% better than C0. Therefore the first run does **not** establish that lagged/SOBI information caused the gain. Most of the benefit may currently come from the simpler act of forcing the two complex latent units not to carry the same second-order information.

That failed attribution is useful. The next attack is to build source pairs where lag-0 covariance is deliberately uninformative but lag spectra differ. Then C0 should lose and SOBI should have a real opportunity to earn the improvement.

## Why this connects to the papers

AMUSE/SOBI show that hidden time-series components can be identifiable from their different autocovariance structures rather than from non-Gaussianity alone. That suggests a learnable latent should be tested against **families of lagged covariance matrices**, not just its instantaneous cloud.

iVAEar pushes the idea further: nonlinear source coordinates become identifiable only when extra temporal/nonstationary structure is supplied under explicit assumptions. This is the serious model to compare against once Tuesday leaves linear/second-order toy worlds.

DDICA shows that a neural unmixing map can be trained directly with an independence/total-correlation objective, but it does not remove the general nonlinear-ICA identifiability problem. For Tuesday it is an empirical attacker, not a license to call any low-TC latent a recovered cause.

## What to build next

1. Replace the synthetic MLP visual encoder/decoder with a **small causal waist inserted into UnitVAE4**.
2. Keep the Stable Video Diffusion VAE as teacher initially, but do not force the causal waist itself to imitate all 16,384 teacher-latent coordinates.
3. Compare A/B/C on webcam sequences with controlled interventions: head motion, illumination, camera motion and patch occlusion.
4. Add a strict lag-information attacker: source processes with equal instantaneous covariance but distinct spectra/autocovariance.
5. Only after that try total-correlation and identifiable nonlinear objectives.
6. The practical regeneration test is dropout/occlusion: can the dynamics prior continue a latent cause and let the decoder restore pixels that are temporarily unobserved?

The claim target is intentionally narrow:

> A temporally structured latent coordinate system can make missing visual information more recoverable from learned dynamics than an equally small reconstruction/prediction latent without that structure.

Anything stronger has to be earned by the attackers.
