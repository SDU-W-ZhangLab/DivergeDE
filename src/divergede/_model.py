"""Numerical building blocks for the DivergeDE model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate, optimize
from scipy.special import digamma, expit, gammaln, logsumexp, polygamma

EPS = 1e-12
MIN_MU = 1e-9
MAX_ETA = 30.0
LOG_CLIP = 50.0
DELTA_MIN = -5.0
DELTA_MAX = 5.0
R_MIN = 1e-4
R_MAX = 1e6
BRANCH_TRANSITION_SLOPE = 12.0


@dataclass(slots=True)
class NullFit:
    beta: np.ndarray
    r: float
    loglik: float
    converged: bool
    message: str


@dataclass(slots=True)
class AlternativeFit:
    tau: float
    delta1: float
    delta2: float
    r: float
    loglik: float
    n_iter: int
    converged: bool
    message: str


def gate(dt: np.ndarray, kappa: float) -> np.ndarray:
    """Return the smooth post-tau branch activation."""
    dt = np.asarray(dt, dtype=float)
    sig = expit(np.clip(float(kappa) * dt, -LOG_CLIP, LOG_CLIP))
    return 2.0 * np.maximum(0.0, sig - 0.5)


def nb_logpmf(y: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    """NB2 log PMF with Var(Y)=mu+mu^2/r."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), MIN_MU, np.inf)
    r = float(np.clip(r, R_MIN, R_MAX))
    return (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + y * (np.log(mu) - np.log(mu + r))
        + r * (np.log(r) - np.log(mu + r))
    )


def _nb_dmu(y: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    mu = np.clip(np.asarray(mu, dtype=float), MIN_MU, np.inf)
    return np.asarray(y, dtype=float) / mu - (np.asarray(y, dtype=float) + r) / (mu + r)


def _nb_d2mu(y: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), MIN_MU, np.inf)
    return -y / (mu**2) + (y + r) / ((mu + r) ** 2)


def _nb_dr(y: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), MIN_MU, np.inf)
    return digamma(y + r) - digamma(r) + np.log(r) + 1.0 - np.log(mu + r) - (y + r) / (mu + r)


def _nb_d2r(y: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), MIN_MU, np.inf)
    return (
        polygamma(1, y + r)
        - polygamma(1, r)
        + 1.0 / r
        - 1.0 / (mu + r)
        + (y - mu) / ((mu + r) ** 2)
    )


def make_basis(t: np.ndarray, df: int) -> tuple[np.ndarray, dict[str, object]]:
    """Build the cubic B-spline basis used by H0."""
    t = np.asarray(t, dtype=float)
    degree = 3
    n_basis = max(int(df), degree + 1)
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    if not t_max > t_min:
        raise ValueError("Pseudotime must vary to build a spline basis.")
    n_inner = n_basis - degree - 1
    if n_inner:
        probs = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
        inner = np.quantile(t, probs)
    else:
        inner = np.empty(0, dtype=float)
    knots = np.concatenate(
        [np.repeat(t_min, degree + 1), np.asarray(inner), np.repeat(t_max, degree + 1)]
    )
    spec: dict[str, object] = {"degree": degree, "knots": knots, "n_basis": n_basis}
    return evaluate_basis(t, spec), spec


def evaluate_basis(t: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    """Evaluate a saved B-spline specification."""
    t = np.asarray(t, dtype=float)
    degree = int(spec["degree"])
    knots = np.asarray(spec["knots"], dtype=float)
    n_basis = int(spec["n_basis"])
    basis = np.zeros((t.size, n_basis), dtype=float)
    for index in range(n_basis):
        coef = np.zeros(n_basis, dtype=float)
        coef[index] = 1.0
        basis[:, index] = interpolate.BSpline(knots, coef, degree, extrapolate=True)(t)
    return basis


def fit_null(
    y: np.ndarray,
    X: np.ndarray,
    log_size_factor: np.ndarray,
    max_iter: int,
    likelihood_tolerance: float,
) -> NullFit:
    """Fit the unpenalized H0 spline and gene-specific r."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    log_size_factor = np.asarray(log_size_factor, dtype=float)
    beta0, *_ = np.linalg.lstsq(X, np.log(y + 0.1) - log_size_factor, rcond=None)
    theta0 = np.concatenate([beta0, np.array([0.0])])
    log_r_bounds = (float(np.log(R_MIN)), float(np.log(R_MAX)))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = theta[:-1]
        r = float(np.exp(np.clip(theta[-1], *log_r_bounds)))
        eta = np.clip(log_size_factor + X @ beta, -MAX_ETA, MAX_ETA)
        mu = np.exp(eta)
        loglik = float(np.sum(nb_logpmf(y, mu, r)))
        grad_beta = X.T @ (_nb_dmu(y, mu, r) * mu)
        grad_log_r = r * float(np.sum(_nb_dr(y, mu, r)))
        return -loglik, -np.concatenate([grad_beta, np.array([grad_log_r])])

    result = optimize.minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, None)] * X.shape[1] + [log_r_bounds],
        options={"maxiter": int(max_iter), "ftol": float(likelihood_tolerance)},
    )
    beta = np.asarray(result.x[:-1], dtype=float)
    r = float(np.exp(np.clip(result.x[-1], *log_r_bounds)))
    mu = np.exp(np.clip(log_size_factor + X @ beta, -MAX_ETA, MAX_ETA))
    loglik = float(np.sum(nb_logpmf(y, mu, r)))
    at_boundary = r <= R_MIN * (1.0 + 1e-6) or r >= R_MAX * (1.0 - 1e-6)
    converged = bool(result.success and np.isfinite(loglik) and not at_boundary)
    message = "r reached its numerical boundary" if at_boundary else str(result.message)
    return NullFit(beta=beta, r=r, loglik=loglik, converged=converged, message=message)


def branch_log_weights(t: np.ndarray, tau: float, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return tau-dependent soft branch log weights."""
    smooth = expit(np.clip(BRANCH_TRANSITION_SLOPE * (np.asarray(t) - float(tau)), -LOG_CLIP, LOG_CLIP))
    q1 = np.clip(0.5 + smooth * (probabilities[:, 0] - 0.5), EPS, 1.0 - EPS)
    q2 = np.clip(0.5 + smooth * (probabilities[:, 1] - 0.5), EPS, 1.0 - EPS)
    return np.log(q1), np.log(q2)


def branch_means(
    mu0: np.ndarray,
    t: np.ndarray,
    tau: float,
    delta1: float,
    delta2: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    activation = gate(np.asarray(t) - float(tau), kappa)
    mu1 = mu0 * np.exp(np.clip(float(delta1) * activation, -LOG_CLIP, LOG_CLIP))
    mu2 = mu0 * np.exp(np.clip(float(delta2) * activation, -LOG_CLIP, LOG_CLIP))
    return activation, mu1, mu2


def observed_loglik(
    y: np.ndarray,
    mu0: np.ndarray,
    t: np.ndarray,
    probabilities: np.ndarray,
    tau: float,
    delta1: float,
    delta2: float,
    r: float,
    kappa: float,
) -> float:
    _, mu1, mu2 = branch_means(mu0, t, tau, delta1, delta2, kappa)
    log_q1, log_q2 = branch_log_weights(t, tau, probabilities)
    terms = np.vstack([log_q1 + nb_logpmf(y, mu1, r), log_q2 + nb_logpmf(y, mu2, r)])
    return float(np.sum(logsumexp(terms, axis=0)))


def e_step(
    y: np.ndarray,
    mu0: np.ndarray,
    t: np.ndarray,
    probabilities: np.ndarray,
    tau: float,
    delta1: float,
    delta2: float,
    r: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, mu1, mu2 = branch_means(mu0, t, tau, delta1, delta2, kappa)
    log_q1, log_q2 = branch_log_weights(t, tau, probabilities)
    a1 = log_q1 + nb_logpmf(y, mu1, r)
    a2 = log_q2 + nb_logpmf(y, mu2, r)
    normalizer = logsumexp(np.vstack([a1, a2]), axis=0)
    return np.exp(a1 - normalizer), np.exp(a2 - normalizer)


def _newton_ascent(
    function,
    initial: float,
    lower: float,
    upper: float,
    step_clip: float = 1.0,
) -> float:
    x = float(np.clip(initial, lower, upper))
    value, gradient, hessian = function(x)
    for _ in range(30):
        if abs(gradient) <= 1e-8 * (1.0 + abs(value)):
            break
        step = -gradient / hessian if np.isfinite(hessian) and abs(hessian) > 1e-12 else np.sign(gradient)
        step = float(np.clip(step, -step_clip, step_clip))
        improved = False
        for _ in range(20):
            candidate = float(np.clip(x + step, lower, upper))
            candidate_value, candidate_gradient, candidate_hessian = function(candidate)
            if np.isfinite(candidate_value) and candidate_value >= value - 1e-12:
                x, value = candidate, candidate_value
                gradient, hessian = candidate_gradient, candidate_hessian
                improved = True
                break
            step *= 0.5
        if not improved or abs(step) <= 1e-8 * (1.0 + abs(x)):
            break
    return x


def update_delta(y: np.ndarray, mu0: np.ndarray, activation: np.ndarray, weights: np.ndarray, r: float, initial: float) -> float:
    """Update one branch effect with fixed responsibilities."""
    if float(np.max(np.abs(activation))) <= EPS:
        return float(np.clip(initial, DELTA_MIN, DELTA_MAX))

    def objective(delta: float) -> tuple[float, float, float]:
        mu = mu0 * np.exp(np.clip(delta * activation, -LOG_CLIP, LOG_CLIP))
        first = _nb_dmu(y, mu, r)
        second = _nb_d2mu(y, mu, r)
        value = float(np.sum(weights * nb_logpmf(y, mu, r)))
        gradient = float(np.sum(weights * first * mu * activation))
        hessian = float(
            np.sum(weights * (second * (mu * activation) ** 2 + first * mu * activation**2))
        )
        return value, gradient, hessian

    return _newton_ascent(objective, initial, DELTA_MIN, DELTA_MAX)


def update_r(
    y: np.ndarray,
    mu1: np.ndarray,
    mu2: np.ndarray,
    weights1: np.ndarray,
    weights2: np.ndarray,
    initial: float,
) -> float:
    """Update the shared gene-specific r on the log scale."""
    lower, upper = float(np.log(R_MIN)), float(np.log(R_MAX))

    def objective(log_r: float) -> tuple[float, float, float]:
        r = float(np.exp(np.clip(log_r, lower, upper)))
        value = float(np.sum(weights1 * nb_logpmf(y, mu1, r) + weights2 * nb_logpmf(y, mu2, r)))
        gradient_r = float(np.sum(weights1 * _nb_dr(y, mu1, r) + weights2 * _nb_dr(y, mu2, r)))
        hessian_r = float(np.sum(weights1 * _nb_d2r(y, mu1, r) + weights2 * _nb_d2r(y, mu2, r)))
        return value, r * gradient_r, r * gradient_r + r**2 * hessian_r

    log_r = _newton_ascent(objective, np.log(np.clip(initial, R_MIN, R_MAX)), lower, upper)
    return float(np.exp(log_r))


def update_tau(
    y: np.ndarray,
    mu0: np.ndarray,
    t: np.ndarray,
    probabilities: np.ndarray,
    weights1: np.ndarray,
    weights2: np.ndarray,
    delta1: float,
    delta2: float,
    r: float,
    kappa: float,
    bounds: tuple[float, float],
) -> float:
    """Update tau by maximizing the unpenalized complete-data objective."""

    def negative_q(tau: float) -> float:
        _, mu1, mu2 = branch_means(mu0, t, tau, delta1, delta2, kappa)
        log_q1, log_q2 = branch_log_weights(t, tau, probabilities)
        value = np.sum(
            weights1 * (log_q1 + nb_logpmf(y, mu1, r))
            + weights2 * (log_q2 + nb_logpmf(y, mu2, r))
        )
        return -float(value)

    result = optimize.minimize_scalar(
        negative_q,
        method="bounded",
        bounds=bounds,
        options={"xatol": 1e-6, "maxiter": 100},
    )
    return float(np.clip(result.x, *bounds))


def quick_start(
    y: np.ndarray,
    mu0: np.ndarray,
    t: np.ndarray,
    probabilities: np.ndarray,
    tau: float,
    r0: float,
    kappa: float,
) -> tuple[float, float, float, float, float]:
    """Estimate unbiased branch effects before scoring one tau grid point."""
    delta1 = 0.0
    delta2 = 0.0
    r = float(r0)
    for _ in range(2):
        weights1, weights2 = e_step(y, mu0, t, probabilities, tau, delta1, delta2, r, kappa)
        activation = gate(t - tau, kappa)
        delta1 = update_delta(y, mu0, activation, weights1, r, delta1)
        delta2 = update_delta(y, mu0, activation, weights2, r, delta2)
        _, mu1, mu2 = branch_means(mu0, t, tau, delta1, delta2, kappa)
        r = update_r(y, mu1, mu2, weights1, weights2, r)
    score = observed_loglik(y, mu0, t, probabilities, tau, delta1, delta2, r, kappa)
    return score, float(tau), delta1, delta2, r


def fit_alternative_start(
    y: np.ndarray,
    mu0: np.ndarray,
    t: np.ndarray,
    probabilities: np.ndarray,
    start: tuple[float, float, float, float, float],
    kappa: float,
    tau_bounds: tuple[float, float],
    max_iter: int,
    likelihood_tolerance: float,
    parameter_tolerance: float,
) -> AlternativeFit:
    """Run one unpenalized ECM start with monotonicity protection."""
    _, tau, delta1, delta2, r = start
    loglik = observed_loglik(y, mu0, t, probabilities, tau, delta1, delta2, r, kappa)
    message = "maximum iterations reached"
    converged = False

    for iteration in range(1, int(max_iter) + 1):
        old = np.array([tau, delta1, delta2, np.log(r)], dtype=float)
        weights1, weights2 = e_step(y, mu0, t, probabilities, tau, delta1, delta2, r, kappa)
        activation = gate(t - tau, kappa)
        proposed_delta1 = update_delta(y, mu0, activation, weights1, r, delta1)
        proposed_delta2 = update_delta(y, mu0, activation, weights2, r, delta2)
        _, mu1, mu2 = branch_means(mu0, t, tau, proposed_delta1, proposed_delta2, kappa)
        proposed_r = update_r(y, mu1, mu2, weights1, weights2, r)
        proposed_tau = update_tau(
            y,
            mu0,
            t,
            probabilities,
            weights1,
            weights2,
            proposed_delta1,
            proposed_delta2,
            proposed_r,
            kappa,
            tau_bounds,
        )
        proposed = np.array([proposed_tau, proposed_delta1, proposed_delta2, np.log(proposed_r)])
        proposed_loglik = observed_loglik(
            y,
            mu0,
            t,
            probabilities,
            proposed_tau,
            proposed_delta1,
            proposed_delta2,
            proposed_r,
            kappa,
        )

        if not np.isfinite(proposed_loglik) or proposed_loglik < loglik - 1e-8 * (1.0 + abs(loglik)):
            accepted = False
            for fraction in 0.5 ** np.arange(1, 13):
                candidate = old + fraction * (proposed - old)
                candidate_r = float(np.exp(candidate[3]))
                candidate_loglik = observed_loglik(
                    y,
                    mu0,
                    t,
                    probabilities,
                    candidate[0],
                    candidate[1],
                    candidate[2],
                    candidate_r,
                    kappa,
                )
                if np.isfinite(candidate_loglik) and candidate_loglik >= loglik - 1e-10:
                    proposed = candidate
                    proposed_tau, proposed_delta1, proposed_delta2 = map(float, candidate[:3])
                    proposed_r = candidate_r
                    proposed_loglik = candidate_loglik
                    accepted = True
                    break
            if not accepted:
                message = "observed log-likelihood decreased after backtracking"
                break

        relative_loglik = abs(proposed_loglik - loglik) / (1.0 + abs(loglik))
        relative_parameters = float(np.max(np.abs(proposed - old) / (1.0 + np.abs(old))))
        tau, delta1, delta2, r = (
            float(proposed_tau),
            float(proposed_delta1),
            float(proposed_delta2),
            float(proposed_r),
        )
        loglik = float(proposed_loglik)
        if relative_loglik < likelihood_tolerance and relative_parameters < parameter_tolerance:
            converged = True
            message = "converged"
            break

    at_boundary = (
        abs(delta1 - DELTA_MIN) <= 1e-6
        or abs(delta1 - DELTA_MAX) <= 1e-6
        or abs(delta2 - DELTA_MIN) <= 1e-6
        or abs(delta2 - DELTA_MAX) <= 1e-6
        or r <= R_MIN * (1.0 + 1e-6)
        or r >= R_MAX * (1.0 - 1e-6)
    )
    if at_boundary:
        converged = False
        message = "a branch effect or r reached its numerical boundary"
    return AlternativeFit(tau, delta1, delta2, r, loglik, iteration, converged, message)

