
# Rayleigh–Rice mixture estimation

Fits a two-component mixture to nonnegative magnitudes, such as those in a change vector magnitude image. It implements the updates in Eq. (20) of [1], and the minimum-error Bayes classifier from the same paper.

The EM parameter estimation method and Bayesian classifier are described and applied to remote sensing multispectral images in this [Research Paper](https://massimozanetti.github.io/assets/pdf/publications/zanetti2015rayleigh.pdf).



## Installation

Place `rayleigh_rice.py` next to your script or somewhere on your Python import path. Install NumPy 1.22 or later and SciPy.

```bash
python -m pip install "numpy>=1.22" scipy
```



## Fit and classify

This example generates a 100 × 200 magnitude image with a Rayleigh background and a Rice change component. The fit uses only the magnitudes. For real data, pass your magnitude image in place of `magnitudes`.

```python
import numpy as np

from rayleigh_rice import RayleighRiceMixture

rng = np.random.default_rng(42)
shape = (100, 200)
alpha, b, nu, sigma = 0.72, 1.2, 4.8, 0.9

changed = rng.random(shape) >= alpha
n_changed = np.count_nonzero(changed)
magnitudes = rng.rayleigh(scale=b, size=shape)
magnitudes[changed] = np.hypot(
    rng.normal(loc=nu, scale=sigma, size=n_changed),
    rng.normal(loc=0, scale=sigma, size=n_changed),
)

model = RayleighRiceMixture().fit(magnitudes)
change_map = model.predict(magnitudes)

print("Estimated parameters", model.params_)
print("Converged", model.converged_, "Iterations", model.n_iter_)
print("Class-switch thresholds", model.decision_thresholds())
print("Change map shape", change_map.shape)
```

`fit()` pools magnitudes from arrays of any shape. The data must be real, finite and nonnegative, with at least two distinct positive values. Zeros use the right-hand limit obtained by removing the common magnitude factor from both densities. Exclude pixels that encode missing data before fitting and classify the valid pixels separately. Predictions are `uint8` with the same shape as their input.



## Model and fitted parameters

The mixture density is

$$
p(x) = \alpha\,f_{\mathrm{Rayleigh}}(x;b) + (1-\alpha)\,f_{\mathrm{Rice}}(x;\nu,\sigma).
$$

`model.params_` returns `(alpha, b, nu, sigma)`:

| Parameter | Meaning |
|---|---|
| `alpha` | Mixture weight of the Rayleigh component (unchanged) |
| `b` | Rayleigh scale |
| `nu` | Rice noncentrality |
| `sigma` | Rice scale |

The scales, noncentrality and decision thresholds are in the original magnitude units.



## Bayes rule and class switches

`predict(x)` is an alias for `bayes_decision_rule(x)`. With equal misclassification costs, it chooses the component with the larger prior-weighted density. The outputs are `0` for Rayleigh (unchanged) and `1` for Rice (changed). Ties go to Rayleigh. At zero magnitude the comparison uses the right-hand limit of the density ratio.

`decision_thresholds()` returns a sorted tuple of the **positive magnitudes where the predicted class changes**. At a switch the weighted densities are equal and their order reverses across the boundary. A point where the curves touch and the same class wins on both sides contributes no threshold. The tuple can contain zero, one or two values.

```python
model = RayleighRiceMixture.from_parameters((0.65, 2.0, 3.0, 0.5))
print(model.decision_thresholds())    # Approximately (2.58, 3.73)
print(model.predict([1.0, 3.0, 5.0]))  # [0 1 0]
```

| Example magnitude | Predicted class |
|---|---|
| 1.0, below both thresholds | Unchanged |
| 3.0, between the thresholds | Changed |
| 5.0, above both thresholds | Unchanged |

In this example the Rice component wins in the middle and the Rayleigh component has the heavier upper tail. With a single threshold, the direction of the switch depends on the fitted parameters. Use `predict()` to assign the labels.



## Initialization and convergence

- `RayleighRiceMixture(max_iter=1000, tol=1e-8, min_scale=1e-8)` controls the iteration cap, the absolute tolerance on the change in average observed log likelihood, and the minimum scale as a fraction of the maximum training magnitude.
- `fit(x, initial=(alpha, b, nu, sigma))` accepts a starting point in the original magnitude units. Require `0 < alpha < 1`, positive scales and `nu >= 0`. A starting value of `nu=0` keeps the Rice component at its Rayleigh limit.
- Automatic initialization starts from a split based on the largest gap between sample quantiles. For highly imbalanced splits it checks alternative seeds on a small sample before fitting the full dataset.
- `n_iter_` counts full-data updates, and `objective_history_` records the average observed log likelihood up to an additive data-only constant, starting at iteration zero. The MATLAB script uses relative change in its expected complete-data log criterion for stopping.
- `converged_` indicates that the tolerance was met. Strongly overlapping components can yield slow or initialization-sensitive fits, including when the convergence criterion is satisfied.



## Reference

[1] Zanetti, M., Bovolo, F., and Bruzzone, L. (2015). “Rayleigh–Rice Mixture Parameter Estimation via EM Algorithm for Change Detection in Multispectral Images.” *IEEE Transactions on Image Processing*, 24(12), 5004–5016.
