"""Rayleigh–Rice mixture estimation and minimum-error classification.

The fixed-point updates follow Eq. (20) of Zanetti, Bovolo and Bruzzone,
IEEE Transactions on Image Processing 24(12), 5004–5016 (2015), and the
accompanying MATLAB implementation by Massimo Zanetti (2017).
Requires NumPy 1.22+ and SciPy.
"""

from numbers import Integral

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import i0e, i1e


_FLOAT_EPS = np.finfo(float).eps
_ROOT_ATOL = np.finfo(float).tiny


def _magnitudes(x):
    values = np.asarray(x)
    if np.iscomplexobj(values):
        raise ValueError("Magnitudes must be real.")
    values = values.astype(float, copy=False)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Magnitudes must be finite and nonnegative.")
    return values


def _parameters(parameters):
    params = np.asarray(parameters, dtype=float)
    if params.shape != (4,) or not np.all(np.isfinite(params)):
        raise ValueError("Supply four finite parameters (alpha, b, nu, sigma).")
    alpha, b, nu, sigma = params
    if not (0 < alpha < 1 and b > 0 and sigma > 0 and nu >= 0):
        raise ValueError("Require 0 < alpha < 1, b > 0, sigma > 0 and nu >= 0.")
    return tuple(map(float, params))


def _weighted_sum(weights, values):
    # A single-threaded reduction avoids BLAS thread overhead on large images.
    return np.einsum("i,i->", weights, values)


def _bessel_ratio(z, scaled_i0):
    ratio = i1e(z) / scaled_i0
    complement = 1 - ratio
    if z.max() > 1e6:
        large = z > 1e6
        inverse = 1 / z[large]
        # Preserve the residual when I1/I0 rounds to 1 in a narrow Rice component.
        complement[large] = inverse * (0.5 + inverse / 8)
        ratio[large] = 1 - complement[large]
    return ratio, complement


class RayleighRiceMixture:
    """Fit alpha * Rayleigh(b) + (1 - alpha) * Rice(nu, sigma).

    Both scales and the Rice noncentrality use the input's units. The
    constructor accepts an iteration cap, an absolute tolerance on the change
    in average log likelihood, and a scale floor as a fraction of the maximum
    training magnitude. The floor regularizes nearly collapsed components.

    After fitting, params_ is (alpha, b, nu, sigma), n_iter_ is the number
    of updates, and converged_ reports whether the tolerance was met.
    objective_history_ contains the average log likelihood, up to additive
    data-only terms, starting at iteration zero.
    """

    def __init__(self, *, max_iter=1000, tol=1e-8, min_scale=1e-8):
        if not isinstance(max_iter, Integral) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive.")
        if not _FLOAT_EPS <= min_scale < 1:
            raise ValueError("min_scale must be in [machine epsilon, 1).")
        self.max_iter = max_iter
        self.tol = tol
        self.min_scale = min_scale

    @classmethod
    def from_parameters(cls, parameters):
        """Construct a model for classification using existing parameters."""
        model = cls()
        model.params_ = _parameters(parameters)
        return model

    @staticmethod
    def _rice_terms(x, nu, sigma):
        delta = x - nu
        z = (nu / sigma**2) * x
        scaled_i0 = i0e(z)
        log_kernel = np.log(scaled_i0) - 0.5 * (delta / sigma) ** 2
        return delta, z, scaled_i0, log_kernel

    def _initialize(self, x, x2, *, split_at=None):
        # MATLAB's largest gap between 100 quantiles; use the median when
        # the split is flat or a side has too few samples or no positive values.
        if split_at is None:
            quantiles = np.quantile(x, (np.arange(100) + 0.5) / 100, method="hazen")
            gaps = np.diff(quantiles)
            split = x <= quantiles[np.argmax(gaps)]
            min_count = max(2, int(0.01 * x.size))
            if (np.count_nonzero(gaps == gaps.max()) != 1
                    or min(split.sum(), (~split).sum()) < min_count
                    or not np.any(x2[split])):
                split = x <= np.median(x)
        else:
            split = x <= split_at
        if not split.any() or split.all() or not np.any(x2[split]):
            # Keep both components viable even if a mass of tied values makes
            # a threshold split unusable.
            return 0.5, max(np.sqrt(x2.mean() / 2), self.min_scale), 0.8, 0.3

        alpha = split.mean()
        b = max(np.sqrt(x2[split].mean() / 2), self.min_scale)
        upper = x[~split]
        nu_start = upper.mean()
        sigma_start = max(upper.std(), self.min_scale)

        def rice_objective(parameters):
            nu, log_sigma = parameters
            sigma = np.exp(log_sigma)
            delta, z, scaled_i0, log_kernel = self._rice_terms(upper, nu, sigma)
            _, one_minus_ratio = _bessel_ratio(z, scaled_i0)
            residual = delta**2 + 2 * nu * upper * one_minus_ratio
            value = 2 * log_sigma - log_kernel.mean()
            gradient = [
                (upper * one_minus_ratio - delta).mean() / sigma**2,
                2 - residual.mean() / sigma**2,
            ]
            return value, gradient

        start = (nu_start, np.log(sigma_start))
        # A positive seed avoids the absorbing nu=0 fixed point when the
        # partition's Rice MLE is at the Rayleigh boundary.
        min_nu = 0.01 * nu_start
        result = minimize(
            rice_objective, start, jac=True, method="L-BFGS-B",
            bounds=((min_nu, 1.0), (np.log(self.min_scale), 0.0)),
            options={"maxiter": 50},
        )
        nu, log_sigma = result.x if np.isfinite(result.fun) else start
        return alpha, b, nu, np.exp(log_sigma)

    def _screen_initialization(self, x, x2, primary):
        # For an imbalanced split, compare seeds on a small reproducible
        # subset in case the largest quantile gap isolates a few tail values.
        if 0.1 <= primary[0] <= 0.9 or x.size < 100:
            return primary
        if x.size > 8192:
            indices = np.random.default_rng(0).choice(x.size, 8192, replace=False)
            x, x2 = x[indices], x2[indices]
        pilot_steps = min(20, self.max_iter)
        best = primary
        _, history, _ = self._iterate(x, x2, primary, pilot_steps)
        best_score = history[-1]
        for threshold in np.quantile(x, (0.25, 0.5)):
            candidate = self._initialize(x, x2, split_at=threshold)
            _, history, _ = self._iterate(x, x2, candidate, pilot_steps)
            if history[-1] > best_score:
                best = candidate
                best_score = history[-1]
        return best

    def fit(self, x, *, initial=None):
        """Fit unlabeled magnitudes of any shape and return self.

        Supply initial=(alpha, b, nu, sigma) to bypass automatic initialization.
        Initial nu=0 keeps the Rice component at its Rayleigh limit. A fit
        requires at least two samples and two distinct positive magnitudes.
        Zeros use the continuous limit obtained by factoring the common x out
        of both densities. Remove zeros beforehand if they encode missing data.
        """
        values = _magnitudes(x).ravel()
        if values.size < 2:
            raise ValueError("At least two magnitudes are required.")
        scale = values.max()
        if not np.any((values > 0) & (values < scale)):
            raise ValueError("Provide at least two distinct positive magnitudes.")
        x = values / scale
        x2 = x**2
        if initial is None:
            params = self._initialize(x, x2)
            params = self._screen_initialization(x, x2, params)
        else:
            alpha, b, nu, sigma = _parameters(initial)
            params = (alpha, max(b / scale, self.min_scale),
                      nu / scale, max(sigma / scale, self.min_scale))
            if not np.all(np.isfinite(params)):
                raise ValueError("Initial parameters are out of range for the data.")

        params, history, converged = self._iterate(x, x2, params, self.max_iter)
        alpha, b, nu, sigma = params
        self.params_ = _parameters((alpha, b * scale, nu * scale, sigma * scale))
        self.n_iter_ = len(history) - 1
        self.converged_ = converged
        self.objective_history_ = np.asarray(history)
        return self

    def _iterate(self, x, x2, params, max_iter):
        history = []
        converged = False
        for iteration in range(max_iter + 1):
            alpha, b, nu, sigma = params
            log_rayleigh = np.log(alpha) - 2 * np.log(b) - (0.5 / b**2) * x2
            delta, z, scaled_i0, log_kernel = self._rice_terms(x, nu, sigma)
            log_rice = (np.log1p(-alpha) - 2 * np.log(sigma)) + log_kernel
            log_total = np.logaddexp(log_rayleigh, log_rice)
            objective = log_total.mean()
            if not np.isfinite(objective):
                raise FloatingPointError("The mixture objective is nonfinite.")
            history.append(objective)
            if iteration and abs(objective - history[-2]) <= self.tol:
                converged = True
                break
            if iteration == max_iter:
                break

            rayleigh = np.exp(log_rayleigh - log_total)
            rice = np.exp(log_rice - log_total)
            n_rayleigh = rayleigh.sum()
            n_rice = rice.sum()
            if min(n_rayleigh, n_rice) <= 0:
                break

            ratio, one_minus_ratio = _bessel_ratio(z, scaled_i0)
            # Algebraically x² + nu² - 2*x*nu*ratio, with reduced cancellation.
            rice_residual = delta**2 + (2 * nu) * x * one_minus_ratio
            # Both Rice updates use the OLD nu and sigma, as in Eq. (20).
            b_next = np.sqrt(_weighted_sum(rayleigh, x2) / (2 * n_rayleigh))
            nu_next = _weighted_sum(rice, x * ratio) / n_rice
            sigma_next = np.sqrt(_weighted_sum(rice, rice_residual) / (2 * n_rice))
            params = (np.clip(n_rayleigh / x.size, _FLOAT_EPS, 1 - _FLOAT_EPS),
                      max(b_next, self.min_scale), nu_next,
                      max(sigma_next, self.min_scale))

        return params, history, converged

    def _decision_coefficients(self):
        if not hasattr(self, "params_"):
            raise RuntimeError("Fit the mixture or use from_parameters first.")
        alpha, b, nu, sigma = self.params_
        # Express the log posterior odds Rice/Rayleigh in units y=x/sigma.
        ratio = sigma / b
        quadratic = 0.5 * (ratio - 1) * (ratio + 1)
        linear = nu / sigma
        intercept = (np.log1p(-alpha) - np.log(alpha)
                     - 2 * np.log(ratio) - 0.5 * linear**2)
        return sigma, quadratic, linear, intercept

    @staticmethod
    def _log_odds(y, quadratic, linear, intercept):
        if quadratic == 0 and linear == 0:
            return np.broadcast_to(intercept, np.shape(y))
        if quadratic == 0:
            z = linear * y
            return intercept + z + np.log(i0e(z))
        if linear == 0:
            return intercept + (quadratic * y) * y
        z = linear * y
        return intercept + (quadratic * y + linear) * y + np.log(i0e(z))

    def bayes_decision_rule(self, x):
        """Return uint8 labels in x's shape, 0 for Rayleigh and 1 for Rice.

        Uses equal misclassification costs and the fitted mixture priors.
        Ties go to Rayleigh. At x=0 the rule uses the right-hand limit.
        """
        x = _magnitudes(x)
        sigma, quadratic, linear, intercept = self._decision_coefficients()
        return np.asarray(
            self._log_odds(x / sigma, quadratic, linear, intercept) > 0,
            dtype=np.uint8,
        )

    predict = bayes_decision_rule

    def decision_thresholds(self):
        """Return the positive class-switch thresholds in ascending order.

        The tuple can contain zero, one or two thresholds. Use predict() to
        assign the class on each interval. A tangency does not switch classes.
        """
        sigma, quadratic, linear, intercept = self._decision_coefficients()

        def odds(y):
            return float(self._log_odds(y, quadratic, linear, intercept))

        def root(left, right):
            return sigma * brentq(odds, left, right, xtol=_ROOT_ATOL)

        def outer_root(left, sign):
            right = max(1.0, linear, 2 * left)
            while sign * odds(right) < 0:
                right *= 2
                if not np.isfinite(right):
                    raise FloatingPointError("A decision threshold is out of range.")
            return root(left, right)

        if quadratic == 0 and linear == 0:
            return ()
        if quadratic >= 0:
            return (outer_root(0.0, +1),) if intercept < 0 else ()
        if 2 * quadratic + 0.5 * linear**2 <= 0:
            return (outer_root(0.0, -1),) if intercept > 0 else ()

        # I1(z)/(z*I0(z)) decreases from 1/2 to 0. This follows by
        # comparing the decreasing ratios 1/(2(k+1)) of the power-series
        # coefficients of I1(z)/z and I0(z) in z².
        def slope_over_y(y):
            z = linear * y
            ratio = 0.5 if z == 0 else (i1e(z) / i0e(z)) / z
            return 2 * quadratic + linear**2 * ratio

        # I1(z)/I0(z) < 1 bounds the position of the unique maximum.
        peak = brentq(slope_over_y, 0.0, -linear / (2 * quadratic),
                      xtol=_ROOT_ATOL)
        if odds(peak) <= 0:
            return ()
        upper = outer_root(peak, -1)
        return (root(0.0, peak), upper) if intercept < 0 else (upper,)
