import numpy as np
import pytest

from calibration import (
    IdentificationError,
    bootstrap_rc,
    calibrated_values,
    hc1,
    ols,
    rc_fit,
)


def test_omitting_treatment_changes_population_adjustment():
    # Covariance order is D, X, W; the structural treatment coefficient is 0.5.
    covariance = np.array([[1, 1, 1], [1, 2, 2], [1, 2, 3]])
    outcome = covariance @ np.array([0.5, 1, 0])
    omitted = np.array([[1, 0, 0], [0, 0, 2 / 3]])
    included = np.array([[1, 0, 0], [0.5, 0, 0.5]])
    wrong = np.linalg.solve(omitted @ covariance @ omitted.T, omitted @ outcome)
    right = np.linalg.solve(included @ covariance @ included.T, included @ outcome)
    np.testing.assert_allclose(wrong, [1, 0.75])
    np.testing.assert_allclose(right, [0.5, 1])


def test_linear_projection_suffices_without_correct_conditional_mean():
    rng = np.random.default_rng(10)
    d, w = rng.normal(size=(2, 30000))
    x = d + w + w**2
    v = np.column_stack((np.ones(len(d)), d))
    m = calibrated_values(v, w[:, None], x, v, w[:, None])
    y = 0.5 * d + x
    np.testing.assert_allclose(ols(np.column_stack((v, m)), y), [0, 0.5, 1], atol=1e-10)
    assert np.mean((x - m) ** 2) > 1


def test_rank_deficient_ols_is_not_silently_pseudoinverted():
    with pytest.raises(IdentificationError):
        ols(np.ones((10, 2)), np.arange(10))


def test_ridge_preserves_unpenalized_regressor_projection():
    rng = np.random.default_rng(11)
    v = np.column_stack((np.ones(40), rng.normal(size=40)))
    p = rng.normal(size=(40, 60))
    x = 3 * v[:, 1] + rng.normal(size=40)
    m = calibrated_values(v, p, x, v, p, penalty=0.1)
    np.testing.assert_allclose(v.T @ (x - m), 0, atol=1e-10)
    assert np.linalg.norm(p.T @ (x - m)) > 0.1


def test_ridge_matches_full_penalized_normal_equations():
    rng = np.random.default_rng(12)
    v = np.column_stack((np.ones(40), rng.normal(size=40)))
    p = rng.normal(size=(40, 4)) + v[:, 1, None]
    x = rng.normal(size=40)
    residual_p = p - v @ ols(v, p)
    scale = np.sqrt(np.mean(residual_p**2, axis=0))
    design = np.column_stack((v, p / scale))
    penalty = np.diag([0, 0, 4, 4, 4, 4])
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ x)
    m = calibrated_values(v, p, x, v, p, penalty=0.1)
    np.testing.assert_allclose(m, design @ coef, atol=1e-12)


def test_bootstrap_refits_both_samples():
    rng = np.random.default_rng(13)

    def draw(n):
        d, noise, error, measurement = rng.normal(size=(4, n))
        x = d + noise
        v = np.column_stack((np.ones(n), d))
        return v, (x + measurement)[:, None], x, 0.5 * d + x + error

    cal, target = draw(60), draw(500)
    actual = bootstrap_rc(cal, target, np.random.default_rng(14), draws=19)
    boot_rng = np.random.default_rng(14)
    expected = []
    for _ in range(19):
        c = boot_rng.integers(60, size=60)
        u = boot_rng.integers(500, size=500)
        expected.append(
            rc_fit(tuple(a[c] for a in cal), tuple(a[u] for a in target))[0]
        )
    np.testing.assert_allclose(actual, np.quantile(expected, [0.025, 0.975], axis=0))
    _, m = rc_fit(cal, target)
    _, se = hc1(np.column_stack((target[0], m)), target[3])
    assert np.isfinite(se).all()


def test_population_ridge_changes_treatment_even_when_unpenalized():
    from itertools import product

    d, v, u = np.array(list(product([-1.0, 1.0], repeat=3))).T
    x, w = d + v, d + v + u
    design = np.column_stack((np.ones(len(d)), d))
    m = calibrated_values(design, w[:, None], x, design, w[:, None], penalty=0.1)
    coef = ols(np.column_stack((design, m)), 0.5 * d + x)
    np.testing.assert_allclose(coef, [0, 0.4, 1.1], atol=1e-12)


def test_stable_conditional_mean_does_not_transport_linear_projection():
    nodes, weights = np.polynomial.legendre.leggauss(12)
    h, weight = (nodes + 1) / 2, weights / 2
    basis = np.column_stack((np.ones(len(h)), h))
    cal_coef = ols(basis * np.sqrt(weight[:, None]), h**2 * np.sqrt(weight))
    np.testing.assert_allclose(cal_coef, [-1 / 6, 1], atol=1e-12)
    m = basis @ cal_coef
    target_weight = weight * 2 * h
    downstream = np.column_stack((np.ones(len(h)), m))
    coef = ols(
        downstream * np.sqrt(target_weight[:, None]), h**2 * np.sqrt(target_weight)
    )
    np.testing.assert_allclose(coef, [-0.1, 1.2], atol=1e-12)


def test_support_gap_lower_bound_at_midpoint():
    nodes, weights = np.polynomial.legendre.leggauss(12)
    h, weight = 1 + (nodes + 1) / 2, weights / 4
    m0, m1 = h, 2 * h - 1
    midpoint = (m0 + m1) / 2
    risk = np.sum(weight * ((midpoint - m0) ** 2 + (midpoint - m1) ** 2) / 2)
    assert risk == pytest.approx(1 / 24)
