"""Regression calibration with independent labeled and target samples."""

import numpy as np


class IdentificationError(ValueError):
    """The requested unregularized regression has no unique coefficient."""


def ols(design, outcome):
    design = np.asarray(design, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if design.ndim != 2 or len(outcome) != len(design):
        raise ValueError("Incompatible regression arrays")
    if not np.isfinite(design).all() or not np.isfinite(outcome).all():
        raise ValueError("Regression arrays must be finite")
    coef, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank < design.shape[1]:
        raise IdentificationError("Regressor matrix is rank deficient")
    return coef


def hc1(design, outcome):
    coef = ols(design, outcome)
    n, p = design.shape
    if n <= p:
        raise IdentificationError("No residual degrees of freedom")
    residual = outcome - design @ coef
    bread = np.linalg.inv(design.T @ design)
    scores = design * residual[:, None]
    covariance = n / (n - p) * bread @ (scores.T @ scores) @ bread
    return coef, np.sqrt(np.maximum(np.diag(covariance), 0))


def calibrated_values(v_cal, p_cal, x_cal, v_target, p_target, penalty=0.0):
    """Fit X on [V,P], leaving all downstream observed regressors V unpenalized.

    Positive penalty minimizes mean squared residual plus penalty * ||b||^2
    after residualizing P against V and standardizing its residual columns.
    The residualization and scale are learned on the calibration sample only.
    """
    if penalty < 0:
        raise ValueError("penalty must be nonnegative")
    if penalty == 0:
        coef = ols(np.column_stack((v_cal, p_cal)), x_cal)
        return np.column_stack((v_target, p_target)) @ coef
    projection = ols(v_cal, p_cal)
    x_projection = ols(v_cal, x_cal)
    residual_p = p_cal - v_cal @ projection
    residual_x = x_cal - v_cal @ x_projection
    scale = np.sqrt(np.mean(residual_p**2, axis=0))
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    standardized = residual_p / scale
    n, p = standardized.shape
    if p <= n:
        coef = np.linalg.solve(
            standardized.T @ standardized + n * penalty * np.eye(p),
            standardized.T @ residual_x,
        )
    else:
        coef = standardized.T @ np.linalg.solve(
            standardized @ standardized.T + n * penalty * np.eye(n), residual_x
        )
    return v_target @ x_projection + (p_target - v_target @ projection) / scale @ coef


def rc_fit(calibration, target, penalty=0.0):
    """Return coefficients and imputed X; samples are (V, P, X, Y)."""
    v_c, p_c, x_c, _ = calibration
    v_u, p_u, _, y_u = target
    m = calibrated_values(v_c, p_c, x_c, v_u, p_u, penalty)
    return ols(np.column_stack((v_u, m)), y_u), m


def resample(sample, rng):
    indices = rng.integers(len(sample[0]), size=len(sample[0]))
    return tuple(array[indices] for array in sample)


def bootstrap_rc(calibration, target, rng, draws=399):
    """Percentile interval, refitting low-dimensional OLS calibration each time.

    Independent iid samples and a regular, identified, fixed-dimensional model
    are the intended setting. A failed fit raises rather than disappearing.
    """
    if draws < 2:
        raise ValueError("At least two bootstrap draws are required")
    estimates = np.array(
        [
            rc_fit(resample(calibration, rng), resample(target, rng))[0]
            for _ in range(draws)
        ]
    )
    return np.quantile(estimates, [0.025, 0.975], axis=0)
