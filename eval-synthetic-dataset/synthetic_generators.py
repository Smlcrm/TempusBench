"""TempusBench synthetic taskbed: data generators.

Each public function generates the full time series for exactly one synthetic
benchmark task.  Univariate generators return an array of shape ``(T,)``;
multivariate/covariate generators return ``(T, m)`` (column order documented in
the docstring).  All randomness flows through a single ``numpy`` Generator
seeded by the ``seed`` argument, so datasets are exactly reproducible and can
be re-drawn with a new seed for dynamic (refreshable) benchmarks.

Design conventions (see synthetic_tasks.md for the full rationale):

* Default length ``T = 2048``, matching the largest context lengths used by
  TempusBench (recommended split: context 512 / horizon 64 rolling windows).
* Base seasonal period ``P = 24`` (an "hourly/daily" analogue); the long
  period 168 = 24*7 gives a "weekly" analogue.
* A common amplitude scale (signal amplitude ~10, noise sd ~1) is used across
  tasks so that a category comparison varies one property at a time.
* Stochastic recursions (AR, GARCH, Markov switching, ...) are either
  initialised from their stationary distribution or preceded by a burn-in, so
  the delivered sample path is not contaminated by transients.
* Every docstring records the exact data-generating process (DGP) and, where
  instructive, the Bayes-optimal forecast, so model error can be compared to
  the irreducible error.

The ``TASKS`` registry at the bottom maps each task name to its generator,
its category tags, its variate type and its target type; ``generate(name)``
is the single entry point used by the dataset build script.
"""

from __future__ import annotations

import numpy as np

DEFAULT_T = 2048          # default series length
PERIOD = 24               # base seasonal period ("daily" analogue)
LONG_PERIOD = 168         # long seasonal period ("weekly" analogue)
BURN_IN = 512             # burn-in steps for recursive processes


def _rng(seed):
    return np.random.default_rng(seed)


def _t(T):
    """Time index 0, 1, ..., T-1 as float."""
    return np.arange(T, dtype=float)


def _ar1(rng, T, phi, sigma, mu=0.0):
    """Stationary AR(1): x_t = mu + phi*(x_{t-1}-mu) + sigma*eps_t.

    Initialised from the stationary distribution N(mu, sigma^2/(1-phi^2)),
    so no burn-in is needed.  Requires |phi| < 1.
    """
    x = np.empty(T)
    x[0] = mu + rng.standard_normal() * sigma / np.sqrt(1.0 - phi**2)
    eps = rng.standard_normal(T) * sigma
    for t in range(1, T):
        x[t] = mu + phi * (x[t - 1] - mu) + eps[t]
    return x


# ======================================================================
# 1. Baseline / predictability controls
# ======================================================================

def iid_gaussian(T=DEFAULT_T, seed=None, mu=0.0, sigma=1.0):
    """White noise: y_t ~ iid N(mu, sigma^2).

    Control task: there is no exploitable structure.  The Bayes point
    forecast is the constant mu and the Bayes MSE is sigma^2 at every
    horizon.  A model that "finds" structure (forecast variance across
    windows well above 0) is hallucinating.
    """
    rng = _rng(seed)
    return mu + sigma * rng.standard_normal(T)


def noise_free_composite(T=DEFAULT_T, seed=None):
    """Deterministic, noise-free signal: trend + two incommensurate sines.

        y_t = 0.004 t + 6 sin(2 pi t / 24) + 3 sin(2 pi t / 41)

    (24 and 41 are coprime, so the joint pattern repeats only every 984
    steps.)  The Bayes error is exactly zero: this measures a model's
    precision ceiling and its ability to represent superposed periodicities
    without the excuse of noise.  ``seed`` is accepted for interface
    uniformity but unused.
    """
    t = _t(T)
    return 0.004 * t + 6.0 * np.sin(2 * np.pi * t / PERIOD) \
        + 3.0 * np.sin(2 * np.pi * t / 41.0)


def low_snr_seasonal(T=DEFAULT_T, seed=None, amp=1.0, sigma=2.0):
    """Weak sinusoid buried in noise: y_t = amp*sin(2 pi t/24) + sigma*eps_t.

    Signal-to-noise ratio (variance) = amp^2/2 / sigma^2 = 0.125.  With ~85
    full cycles in the context the seasonal component is statistically
    recoverable (periodogram peak >> noise floor), so a good model should
    beat the unconditional mean; a model that gives up and predicts the
    mean loses amp^2/2 of explainable variance.
    """
    rng = _rng(seed)
    t = _t(T)
    return amp * np.sin(2 * np.pi * t / PERIOD) + sigma * rng.standard_normal(T)


def ma1(T=DEFAULT_T, seed=None, theta=0.8, sigma=1.0):
    """Invertible MA(1): y_t = eps_t + theta*eps_{t-1}, eps ~ N(0, sigma^2).

    Memory of exactly one lag: the Bayes forecast is non-trivial at horizon
    1 (E[y_{t+1}|F_t] = theta*eps_t) and exactly 0 for horizons >= 2.  Tests
    both short-memory exploitation and the calibration to *stop* predicting
    beyond the memory of the process.
    """
    rng = _rng(seed)
    eps = rng.standard_normal(T + 1) * sigma
    return eps[1:] + theta * eps[:-1]


# ======================================================================
# 2. Trend / movement
# ======================================================================

def mean_reverting_ar1(T=DEFAULT_T, seed=None, phi=0.8, sigma=1.0):
    """Stationary AR(1), phi=0.8: strongly mean-reverting level.

    Bayes forecast decays geometrically to the mean: E[y_{t+h}|y_t] =
    phi^h y_t.  The canonical "stationary movement" task.
    """
    return _ar1(_rng(seed), T, phi, sigma)


def near_unit_root_ar1(T=DEFAULT_T, seed=None, phi=0.995, sigma=1.0):
    """AR(1) with phi=0.995: stationary but nearly integrated.

    Discriminates models that over-difference (treat as random walk => flat
    forecasts, ignoring slow reversion) from those that over-mean-revert.
    Stationary sd = sigma/sqrt(1-phi^2) ~= 10; mean-reversion half-life
    ln(0.5)/ln(phi) ~= 138 steps, i.e. reversion is visible across a
    512-step context but negligible within a 64-step horizon.
    """
    return _ar1(_rng(seed), T, phi, sigma)


def random_walk(T=DEFAULT_T, seed=None, sigma=1.0):
    """Driftless random walk: y_t = y_{t-1} + sigma*eps_t, y_0 = 0.

    Difference-stationary (unit root).  Bayes forecast is the last observed
    value at every horizon (martingale), with forecast variance sigma^2 h.
    Models should neither extrapolate local drift nor revert to the
    historical mean.
    """
    rng = _rng(seed)
    return np.cumsum(sigma * rng.standard_normal(T))


def random_walk_drift(T=DEFAULT_T, seed=None, drift=0.08, sigma=1.0):
    """Random walk with drift: y_t = drift + y_{t-1} + sigma*eps_t.

    drift = 0.08 makes the drift identifiable from the realisation the model
    actually sees: the sample-mean increment has s.e. sigma/sqrt(T) ~= 0.022,
    so the drift estimate carries an expected t-statistic of ~3.6 (seed-
    dependent in realisation), and the cumulative
    drift (drift*T ~= 164) dominates the walk's typical excursion
    (sigma*sqrt(T) ~= 45).  (An earlier drift of 0.02 was rejected: it was
    statistically unidentifiable from a single path - s.e. of the estimate
    equal to the drift itself - making the task a duplicate of random_walk.)
    Bayes forecast: last value + drift*h.
    """
    rng = _rng(seed)
    return np.cumsum(drift + sigma * rng.standard_normal(T))


def linear_trend(T=DEFAULT_T, seed=None, a=10.0, b=20.0, sigma=1.0):
    """Trend-stationary linear trend: y_t = a + b*(t/T) + sigma*eps_t.

    Deterministic rise of 20 = 20 noise sds over the series.  Unlike the
    random walk with drift, forecast uncertainty does NOT grow with horizon
    (Bayes MSE = sigma^2 for all h) - the diagnostic pair for
    trend- vs difference-stationarity.
    """
    rng = _rng(seed)
    return a + b * (_t(T) / T) + sigma * rng.standard_normal(T)


def exponential_trend(T=DEFAULT_T, seed=None, y0=3.0, growth=10.0, sigma=0.8):
    """Exponential growth: y_t = y0 * growth^(t/T) + sigma*eps_t.

    Level rises 3 -> 30.  Tests out-of-range level extrapolation with
    accelerating increments; linear extrapolation of the context under-
    shoots the horizon.
    """
    rng = _rng(seed)
    return y0 * np.power(growth, _t(T) / T) + sigma * rng.standard_normal(T)


def log_trend(T=DEFAULT_T, seed=None, base=2.0, c=2.0, sigma=0.5):
    """Logarithmic (decelerating) trend: y_t = base + c*ln(1+t) + sigma*eps_t.

    Growth without a finite asymptote but with ever-slowing increments;
    naive linear extrapolation overshoots.  The baseline offset keeps the
    series positive at early t despite additive noise.
    """
    rng = _rng(seed)
    return base + c * np.log1p(_t(T)) + sigma * rng.standard_normal(T)


def power_trend(T=DEFAULT_T, seed=None, base=2.0, scale=10.0, p=0.5,
                sigma=0.5):
    """Power-law ("p-root") trend: y_t = base + scale*(t/T)^p + sigma*eps_t.

    With p=0.5, concave sub-linear growth; increments decay like t^(p-1).
    Intermediate between linear and logarithmic deceleration.  The baseline
    offset keeps the series positive at early t despite additive noise.
    """
    rng = _rng(seed)
    return base + scale * np.power(_t(T) / T, p) + sigma * rng.standard_normal(T)


def logistic_trend(T=DEFAULT_T, seed=None, base=2.0, L=30.0, frac_mid=0.85,
                   frac_width=0.2, sigma=0.5):
    """Sigmoid (saturating) trend: y_t = base + L/(1 + exp(-k(t - t0))) + sigma*eps_t

    with t0 = frac_mid*T and k = 4.4/(frac_width*T) (10%-90% transition
    spans ~frac_width*T steps).  The inflection sits at 0.85*T, so rolling
    evaluation windows successively face acceleration, inflection and
    saturation - the property under test is recognising that apparent
    exponential growth is about to saturate.  The baseline offset keeps the
    series positive before the rise despite additive noise.
    """
    rng = _rng(seed)
    t0 = frac_mid * T
    k = 4.4 / (frac_width * T)
    sig = L / (1.0 + np.exp(-k * (_t(T) - t0)))
    return base + sig + sigma * rng.standard_normal(T)


def piecewise_linear_trend(T=DEFAULT_T, seed=None, frac_break=0.6,
                           b1=15.0, b2=-10.0, sigma=1.0):
    """Broken trend: slope b1/T before the break at frac_break*T, b2/T after
    (continuous at the break), plus N(0, sigma^2) noise.

    A slope *reversal* (up then down).  Tests whether a model conditions on
    the post-break regime instead of the full-context average slope.
    """
    rng = _rng(seed)
    t = _t(T)
    tb = frac_break * T
    trend = np.where(t <= tb, b1 * t / T, b1 * tb / T + b2 * (t - tb) / T)
    return 5.0 + trend + sigma * rng.standard_normal(T)


# ======================================================================
# 3. Seasonality
# ======================================================================

def sinusoidal_seasonal(T=DEFAULT_T, seed=None, amp=10.0, period=PERIOD,
                        sigma=1.0):
    """Pure stationary cycle: y_t = amp*sin(2 pi t/period) + sigma*eps_t.

    The reference seasonality task: high SNR, fixed period, sinusoidal
    shape.  Bayes forecast is the sinusoid itself (MSE sigma^2).
    """
    rng = _rng(seed)
    return amp * np.sin(2 * np.pi * _t(T) / period) + sigma * rng.standard_normal(T)


def multi_seasonal(T=DEFAULT_T, seed=None, sigma=1.0):
    """Two nested periods (daily-in-weekly analogue):

        y_t = 6 sin(2 pi t/24) + 4 sin(2 pi t/168) + sigma*eps_t.

    168 = 7*24, so the short cycle nests in the long one.  Tests whether a
    model separates superposed periodicities; a single-period model leaves
    the amplitude-4 component (variance 8) unexplained.  For the long cycle
    to matter, use context >= 512 (3 weekly cycles).
    """
    rng = _rng(seed)
    t = _t(T)
    return 6.0 * np.sin(2 * np.pi * t / PERIOD) \
        + 4.0 * np.sin(2 * np.pi * t / LONG_PERIOD) \
        + sigma * rng.standard_normal(T)


def nonsinusoidal_seasonal(T=DEFAULT_T, seed=None, kappa=2.0, sigma=1.0):
    """Sharp, asymmetric periodic shape (exponentiated-sine "spike train"):

        s(t) = exp(kappa*[sin(w t) + 0.5 sin(2 w t)]),  w = 2 pi / 24,
        y_t  = 10 * (s(t) - mean(s)) / (max(s) - min(s)) * 2 + sigma*eps_t.

    The pattern is peaked and left-right asymmetric - deliberately far from
    a sinusoid - normalised to amplitude ~10.  Tests seasonal *shape*
    fidelity: a Fourier-truncated or smoothness-biased model rounds off the
    peaks, which the per-step metrics punish.
    """
    rng = _rng(seed)
    t = _t(T)
    w = 2 * np.pi / PERIOD
    s = np.exp(kappa * (np.sin(w * t) + 0.5 * np.sin(2 * w * t)))
    s = (s - s.mean()) / (s.max() - s.min()) * 20.0
    return s + sigma * rng.standard_normal(T)


def trend_seasonal_additive(T=DEFAULT_T, seed=None, sigma=1.0):
    """Non-stationary cyclical, additive composition:

        y_t = 10 + 15*(t/T) + 8 sin(2 pi t/24) + sigma*eps_t.

    Seasonal amplitude is constant while the level rises - the additive
    benchmark, and simultaneously the homoskedastic reference case for the
    noise category (constant conditional variance sigma^2).
    """
    rng = _rng(seed)
    t = _t(T)
    return 10.0 + 15.0 * (t / T) + 8.0 * np.sin(2 * np.pi * t / PERIOD) \
        + sigma * rng.standard_normal(T)


def trend_seasonal_multiplicative(T=DEFAULT_T, seed=None, sigma_rel=0.05):
    """Non-stationary cyclical, multiplicative composition:

        y_t = trend_t * s_t * (1 + sigma_rel*eps_t),
        trend_t = 10*(1 + t/T),   s_t = 1 + 0.4 sin(2 pi t/24).

    Both the seasonal swing and the noise sd scale with the level (the
    series is strictly positive).  The contrast with the additive task
    isolates whether a model infers the composition type; it is also an
    intrinsically level-heteroskedastic noise task.
    """
    rng = _rng(seed)
    t = _t(T)
    trend = 10.0 * (1.0 + t / T)
    s = 1.0 + 0.4 * np.sin(2 * np.pi * t / PERIOD)
    return trend * s * (1.0 + sigma_rel * rng.standard_normal(T))


def damped_seasonal(T=DEFAULT_T, seed=None, amp0=12.0, sigma=1.0):
    """Regressive (decaying-amplitude) cycle:

        y_t = amp0 * exp(-t/tau) * sin(2 pi t/24) + sigma*eps_t,  tau = T/2.

    Amplitude decays 12 -> 12*e^-2 ~= 1.6 across the series.  The model
    must extrapolate the *envelope*, not the last-seen amplitude: repeating
    the final context cycle overshoots the horizon amplitude.
    """
    rng = _rng(seed)
    t = _t(T)
    return amp0 * np.exp(-t / (T / 2.0)) * np.sin(2 * np.pi * t / PERIOD) \
        + sigma * rng.standard_normal(T)


def irregular_period_seasonal(T=DEFAULT_T, seed=None, amp=10.0, sigma=1.0,
                              mod_depth=0.3, mod_period=512.0):
    """Frequency-modulated cycle with smoothly drifting period.

    The instantaneous frequency f(t) = (1/24)*(1 + mod_depth*sin(2 pi t /
    mod_period)) is integrated into a phase (phi_t = 2 pi * cumsum f), and

        y_t = amp * sin(phi_t) + sigma*eps_t.

    The local period therefore truly varies between ~18.5 and ~34 steps.
    (Naively writing sin(2 pi t / p(t)) does NOT do this - the derivative
    of t/p(t) is not 1/p(t) - hence the phase-accumulation construction.)
    Tests phase tracking when the period cannot be assumed constant.
    """
    rng = _rng(seed)
    t = _t(T)
    freq = (1.0 / PERIOD) * (1.0 + mod_depth * np.sin(2 * np.pi * t / mod_period))
    phase = 2 * np.pi * np.cumsum(freq)
    return amp * np.sin(phase) + sigma * rng.standard_normal(T)


def evolving_seasonal(T=DEFAULT_T, seed=None, sigma=1.0):
    """Slowly morphing seasonal shape via drifting Fourier coefficients.

        y_t = sum_{k=1..3} [a_k(t) cos(k w t) + b_k(t) sin(k w t)] + sigma*eps_t

    with w = 2 pi/24.  Each coefficient follows a highly persistent AR(1)
    (phi = 0.997, innovation sd 0.15) around base values a = (0,0,0),
    b = (6,3,1.5) - mean-reverting, so amplitudes stay bounded (unlike a
    random-walk drift, which would wander arbitrarily).  The seasonal shape
    a model should use is the *recent* one, not the context-wide average.
    """
    rng = _rng(seed)
    t = _t(T)
    w = 2 * np.pi / PERIOD
    base_a = np.array([0.0, 0.0, 0.0])
    base_b = np.array([6.0, 3.0, 1.5])
    y = sigma * rng.standard_normal(T)
    for k in range(3):
        a = _ar1(rng, T, 0.997, 0.15, mu=base_a[k])
        b = _ar1(rng, T, 0.997, 0.15, mu=base_b[k])
        y += a * np.cos((k + 1) * w * t) + b * np.sin((k + 1) * w * t)
    return y


# ======================================================================
# 4. Noise / innovation structure
#    (base signal held fixed: sinusoid, so the noise is the only thing
#     varied relative to sinusoidal_seasonal)
# ======================================================================

def _base_signal(T):
    """Common carrier signal for the noise-category tasks."""
    return 10.0 * np.sin(2 * np.pi * _t(T) / PERIOD)


def heteroskedastic_level(T=DEFAULT_T, seed=None, rel=0.08):
    """Level-dependent (multiplicative-style) heteroskedasticity:

        level_t = 12 + 6 sin(2 pi t / 1024)   (slow, NON-monotonic),
        y_t = level_t + 3 sin(2 pi t/24) + rel*level_t*eps_t.

    Noise sd is proportional to the level.  The level is deliberately
    non-monotonic so that sigma ~ level is distinguishable from sigma ~ t
    (with a monotone level the two are confounded).  Probabilistic metrics
    (CRPS/WIS) reward models whose predictive spread tracks the level.
    """
    rng = _rng(seed)
    t = _t(T)
    level = 12.0 + 6.0 * np.sin(2 * np.pi * t / 1024.0)
    return level + 3.0 * np.sin(2 * np.pi * t / PERIOD) \
        + rel * level * rng.standard_normal(T)


def heteroskedastic_time(T=DEFAULT_T, seed=None, sigma0=0.5, sigma1=2.5):
    """Time-driven variance growth: y_t = 10 sin(2 pi t/24) + sigma(t)*eps_t,
    with sigma(t) = sigma0 + (sigma1-sigma0)*(t/T) rising 0.5 -> 2.5.

    Conditional-mean structure is unchanged from the reference sinusoid;
    only the noise level trends.  Point metrics should degrade gracefully
    and predictive intervals should widen with t.
    """
    rng = _rng(seed)
    sig = sigma0 + (sigma1 - sigma0) * (_t(T) / T)
    return _base_signal(T) + sig * rng.standard_normal(T)


def garch_noise(T=DEFAULT_T, seed=None, omega=0.05, alpha=0.1, beta=0.85):
    """GARCH(1,1) with zero conditional mean (volatility clustering):

        y_t = sigma_t * eps_t,   sigma_t^2 = omega + alpha*y_{t-1}^2
                                             + beta*sigma_{t-1}^2.

    alpha+beta = 0.95 < 1 (covariance-stationary, unconditional variance
    omega/(1-alpha-beta) = 1); initialised at the unconditional variance
    with a burn-in.  The Bayes point forecast is 0 - the task is purely
    probabilistic: predictive spread must expand and contract with the
    volatility state.  Excess kurtosis is the visible fingerprint.
    """
    rng = _rng(seed)
    n = T + BURN_IN
    y = np.empty(n)
    var = omega / (1.0 - alpha - beta)
    eps = rng.standard_normal(n)
    y[0] = np.sqrt(var) * eps[0]
    for t in range(1, n):
        var = omega + alpha * y[t - 1] ** 2 + beta * var
        y[t] = np.sqrt(var) * eps[t]
    return y[BURN_IN:]


def heavy_tailed_noise(T=DEFAULT_T, seed=None, df=3.0, scale=1.0):
    """Student-t(3) innovations on the reference sinusoid:

        y_t = 10 sin(2 pi t/24) + scale * t_df / sqrt(df/(df-2)).

    Innovations are standardised to unit variance but have infinite fourth
    moment: rare shocks are ~an order of magnitude larger than Gaussian
    ones.  Tests robustness of context encoding and realistic tail width in
    predictive distributions.
    """
    rng = _rng(seed)
    innov = rng.standard_t(df, size=T) / np.sqrt(df / (df - 2.0))
    return _base_signal(T) + scale * innov


def skewed_noise(T=DEFAULT_T, seed=None, s=0.8):
    """Right-skewed innovations (centred, unit-variance log-normal):

        e_t = (LN(0, s^2) - exp(s^2/2)) / sqrt((exp(s^2)-1) exp(s^2)),
        y_t = 10 sin(2 pi t/24) + e_t.

    Mean-zero but skew ~3.7: the median lies below the mean, so a model
    trained toward the median (MAE-style) biases low on the mean and vice
    versa.  Separates mean- from median-calibrated forecasters.
    """
    rng = _rng(seed)
    raw = rng.lognormal(mean=0.0, sigma=s, size=T)
    m = np.exp(s**2 / 2.0)
    sd = np.sqrt((np.exp(s**2) - 1.0) * np.exp(s**2))
    return _base_signal(T) + (raw - m) / sd


def autocorrelated_noise(T=DEFAULT_T, seed=None, phi=0.8, sigma=0.6):
    """AR(1) errors around a deterministic sinusoid:

        y_t = 10 sin(2 pi t/24) + u_t,   u_t = phi*u_{t-1} + sigma*eps_t.

    The optimal forecast corrects the sinusoid by phi^h times the last
    residual - a model that treats residuals as white noise leaves
    first-lag structure on the table.  (Classic regression-with-ARMA-errors
    setting.)
    """
    rng = _rng(seed)
    return _base_signal(T) + _ar1(rng, T, phi, sigma)


def outlier_contaminated(T=DEFAULT_T, seed=None, p=0.02, out_scale=10.0,
                         sigma=1.0):
    """Additive-outlier contamination of the reference sinusoid:

        y_t = 10 sin(2 pi t/24) + sigma*eps_t + o_t,
        o_t = 0 w.p. 1-p;  +/- U(8, 15)*sigma w.p. p (sign fair-coin).

    Isolated impulses (~2% of points) that carry no information about the
    future.  A robust model ignores them; a fragile context encoding lets a
    single spike distort level/seasonal estimates.
    """
    rng = _rng(seed)
    y = _base_signal(T) + sigma * rng.standard_normal(T)
    mask = rng.random(T) < p
    n_out = int(mask.sum())
    signs = rng.choice([-1.0, 1.0], size=n_out)
    y[mask] += signs * rng.uniform(8.0, 15.0, size=n_out) * sigma
    return y


def measurement_error_rw(T=DEFAULT_T, seed=None, sigma_proc=0.3,
                         sigma_obs=1.0):
    """Local-level state-space model (signal + measurement error):

        x_t = x_{t-1} + sigma_proc * w_t     (latent level, random walk),
        y_t = x_t + sigma_obs * v_t          (noisy observation).

    Signal-to-measurement ratio q = (sigma_proc/sigma_obs)^2 = 0.09.  The
    Bayes forecast is the Kalman-filtered level (equivalently exponential
    smoothing with the steady-state gain ~0.26) - neither the last
    observation (too noisy) nor a long mean (too stale).  This is the
    "measurement error" data-quality task, distinct from process noise.
    """
    rng = _rng(seed)
    x = np.cumsum(sigma_proc * rng.standard_normal(T))
    return x + sigma_obs * rng.standard_normal(T)


# ======================================================================
# 5. Autocorrelation / memory
# ======================================================================

def ar2_pseudocyclic(T=DEFAULT_T, seed=None, phi1=1.5, phi2=-0.9, sigma=1.0):
    """Stationary AR(2) with complex roots (stochastic pseudo-cycles):

        y_t = phi1*y_{t-1} + phi2*y_{t-2} + sigma*eps_t.

    Spectral peak near period 2 pi / arccos(phi1/(2 sqrt(-phi2))) ~= 9.5,
    but the phase diffuses: unlike true seasonality the cycle cannot be
    extrapolated far ahead.  Distinguishes models that infer dynamics from
    those that pattern-match a fixed calendar period.  Burn-in removes the
    zero-initialisation transient.
    """
    rng = _rng(seed)
    n = T + BURN_IN
    y = np.zeros(n)
    eps = rng.standard_normal(n) * sigma
    for t in range(2, n):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + eps[t]
    return y[BURN_IN:]


def long_memory_fgn(T=DEFAULT_T, seed=None, H=0.9):
    """Fractional Gaussian noise with Hurst H=0.9 (long memory).

    Exact simulation via the Hosking/Durbin-Levinson recursion on the fGn
    autocovariance gamma(k) = 0.5*(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H}).
    Autocorrelations decay hyperbolically (~ k^{2H-2} = k^{-0.2}); distant
    context remains informative, and predictability decays much more slowly
    with horizon than for any ARMA process.  Scaled by 3 for a working
    amplitude comparable to other tasks.
    """
    rng = _rng(seed)
    k = np.arange(T, dtype=float)
    gamma = 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H)
                   + np.abs(k - 1) ** (2 * H))
    x = np.empty(T)
    v = gamma[0]
    x[0] = rng.standard_normal() * np.sqrt(v)
    phi_prev = np.zeros(0)
    for t in range(1, T):
        if t == 1:
            kap = gamma[1] / v
        else:
            kap = (gamma[t] - phi_prev @ gamma[t - 1:0:-1]) / v
        phi_new = np.empty(t)
        phi_new[:t - 1] = phi_prev - kap * phi_prev[::-1]
        phi_new[t - 1] = kap
        v = v * (1.0 - kap ** 2)
        x[t] = phi_new @ x[t - 1::-1] + np.sqrt(v) * rng.standard_normal()
        phi_prev = phi_new
    return 3.0 * x


# ======================================================================
# 6. Nonlinearity / deterministic dynamics
# ======================================================================

def setar(T=DEFAULT_T, seed=None, phi_low=0.95, phi_high=0.4, sigma=0.5):
    """Self-exciting threshold AR with asymmetric persistence (threshold 0):

        y_t = 0.95 y_{t-1} + sigma*eps_t   if y_{t-1} <= 0,
        y_t = 0.40 y_{t-1} + sigma*eps_t   if y_{t-1} >  0.

    Both regimes mean-revert to 0 (geometrically ergodic), but negative
    excursions decay slowly while positive ones die out almost immediately
    - the signature sign-asymmetry of threshold models (cf. unemployment
    dynamics).  A single linear AR fits an averaged phi and systematically
    over-predicts persistence above the threshold and under-predicts it
    below, so the task cleanly separates linear from nonlinear models.
    (An earlier design with opposite-signed regime intercepts was rejected:
    it produced a near period-2 flip-flop that a linear AR with a negative
    lag-1 coefficient imitates well.)
    """
    rng = _rng(seed)
    n = T + BURN_IN
    y = np.zeros(n)
    eps = rng.standard_normal(n) * sigma
    for t in range(1, n):
        phi = phi_low if y[t - 1] <= 0.0 else phi_high
        y[t] = phi * y[t - 1] + eps[t]
    return y[BURN_IN:]


def logistic_map(T=DEFAULT_T, seed=None, r=3.9, obs_sigma=0.0):
    """Chaotic logistic map: x_{t+1} = r x_t (1 - x_t), r=3.9, x in (0,1).

        y_t = 10*x_t + obs_sigma*eps_t   (scaled to working amplitude).

    Fully deterministic: one-step dynamics are exactly learnable (a smooth
    quadratic map), but sensitivity to initial conditions (Lyapunov exponent
    ~0.5) makes long horizons intrinsically unpredictable - point accuracy
    must degrade toward the invariant distribution at a known exponential
    rate.  Tests short-horizon nonlinear dynamics *and* long-horizon
    uncertainty growth.  Initial condition drawn U(0.1, 0.9); 200-step
    transient discarded.
    """
    rng = _rng(seed)
    x = rng.uniform(0.1, 0.9)
    for _ in range(200):
        x = r * x * (1.0 - x)
    out = np.empty(T)
    for t in range(T):
        x = r * x * (1.0 - x)
        out[t] = x
    y = 10.0 * out
    if obs_sigma > 0:
        y = y + obs_sigma * rng.standard_normal(T)
    return y


def mackey_glass(T=DEFAULT_T, seed=None, tau=17.0, beta=0.2, gamma=0.1,
                 n_exp=10, dt=0.1, stride=10):
    """Mackey-Glass delay differential equation (chaotic regime, tau=17):

        dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t),

    integrated by Euler with step dt=0.1 and sub-sampled every ``stride``
    steps (sampling interval 1 time unit - the standard benchmark setup).
    History initialised at 1.2 plus a small seeded perturbation; the first
    500 time units are discarded as transient.  Smooth quasi-periodic
    chaotic oscillations: strong short-range predictability with slow
    divergence - the classic nonlinear-forecasting benchmark, complementary
    to the logistic map (continuous & smooth vs discrete & jagged).
    Scaled by 10.
    """
    rng = _rng(seed)
    hist = int(round(tau / dt))
    discard_units = 500
    n_steps = hist + (discard_units + T) * stride
    x = np.empty(n_steps)
    x[:hist] = 1.2 + 0.05 * rng.standard_normal()
    for i in range(hist, n_steps):
        xd = x[i - hist]
        x[i] = x[i - 1] + dt * (beta * xd / (1.0 + xd ** n_exp)
                                - gamma * x[i - 1])
    series = x[hist + discard_units * stride::stride][:T]
    return 10.0 * series


# ======================================================================
# 7. Structural change / regimes
# ======================================================================

def level_shift(T=DEFAULT_T, seed=None, frac_break=0.7, shift=6.0, sigma=1.0):
    """One-off mean shift on the reference sinusoid:

        y_t = 10 sin(2 pi t/24) + shift*1{t >= 0.7T} + sigma*eps_t.

    A permanent +6 sigma level jump.  Post-break, the correct level is the
    new one; models anchored to the full-context mean systematically bias
    low.  (Under rolling windows, early windows also probe behaviour when
    the break sits inside the forecast context at varying depths.)
    """
    rng = _rng(seed)
    t = _t(T)
    return _base_signal(T) + shift * (t >= frac_break * T) \
        + sigma * rng.standard_normal(T)


def variance_shift(T=DEFAULT_T, seed=None, frac_break=0.6, sigma1=1.0,
                   sigma2=2.5):
    """One-off variance break: sigma jumps 1.0 -> 2.5 at 0.6T on the
    reference sinusoid (conditional mean unchanged).

    Point-forecast difficulty is unchanged; predictive-interval width must
    change.  Complements level_shift by breaking the second moment only.
    """
    rng = _rng(seed)
    t = _t(T)
    sig = np.where(t < frac_break * T, sigma1, sigma2)
    return _base_signal(T) + sig * rng.standard_normal(T)


def markov_switching_ar(T=DEFAULT_T, seed=None, p_stay=0.97, sigma=0.5):
    """Recurring regimes: 2-state Markov chain (stay prob 0.97, mean sojourn
    ~33 steps) switching the mean and dynamics of an AR(1):

        state 0:  y_t = -1.5 + 0.5*(y_{t-1}+1.5) + sigma*eps_t
        state 1:  y_t = +1.5 + 0.9*(y_{t-1}-1.5) + sigma*eps_t

    Unlike one-off breaks, regimes recur (~60 switches per series), so the
    regime structure is identifiable from context and the optimal forecast
    is a probability-weighted mixture over future regime paths.  Burn-in
    applied.
    """
    rng = _rng(seed)
    n = T + BURN_IN
    mu = (-1.5, 1.5)
    phi = (0.5, 0.9)
    s = rng.integers(0, 2)
    y = np.zeros(n)
    for t in range(1, n):
        if rng.random() > p_stay:
            s = 1 - s
        y[t] = mu[s] + phi[s] * (y[t - 1] - mu[s]) \
            + sigma * rng.standard_normal()
    return y[BURN_IN:]


# ======================================================================
# 8. Target type / marginal support
# ======================================================================

def binary_latent_ar(T=DEFAULT_T, seed=None, phi=0.9, sigma=0.4, amp=1.5):
    """Binary series from a thresholded latent process:

        z_t = phi*z_{t-1} + sigma*eps_t + amp*sin(2 pi t/24) applied as
        z_t = AR(1)(phi, sigma) + amp*sin(2 pi t/24),
        y_t = 1{z_t > 0}  in {0, 1}.

    Persistence and seasonality make P(y_{t+h}=1 | context) genuinely
    dynamic (roughly 0.1-0.9 over a cycle); the Bayes probability is the
    Gaussian CDF of the predicted latent mean over its predictive sd.
    Tests forecasting on a two-point support where regression-style outputs
    must be interpretable as probabilities/thresholded states.
    """
    rng = _rng(seed)
    z = _ar1(rng, T, phi, sigma) + amp * np.sin(2 * np.pi * _t(T) / PERIOD)
    return (z > 0.0).astype(float)


def ordinal_categorical(T=DEFAULT_T, seed=None, phi=0.9, sigma=0.4, amp=1.5):
    """Ordered 3-category series: the same latent construction as
    binary_latent_ar, cut at thresholds (-0.8, +0.8):

        y_t = 0 if z_t <= -0.8;  1 if -0.8 < z_t <= 0.8;  2 otherwise.

    Covers the 'categorical' target type of the TempusBench taxonomy with
    an ordered state space (unordered categories have no natural error
    metric under MAE/RMSE-style evaluation).
    """
    rng = _rng(seed)
    z = _ar1(rng, T, phi, sigma) + amp * np.sin(2 * np.pi * _t(T) / PERIOD)
    return np.digitize(z, (-0.8, 0.8)).astype(float)


def poisson_counts(T=DEFAULT_T, seed=None):
    """Non-negative counts, canonical log-link Poisson:

        lambda_t = exp(1.5 + 0.4*(t/T) + 0.7 sin(2 pi t/24)),
        y_t ~ Poisson(lambda_t)     (lambda ranges ~2.2 -> ~13.5).

    Equidispersed (Var = mean).  Discreteness, positivity, and the
    mean-variance link distinguish count data from continuous data; the
    Bayes point forecast is lambda_t (mean) or the Poisson median.
    """
    rng = _rng(seed)
    t = _t(T)
    lam = np.exp(1.5 + 0.4 * (t / T) + 0.7 * np.sin(2 * np.pi * t / PERIOD))
    return rng.poisson(lam).astype(float)


def negbin_counts(T=DEFAULT_T, seed=None, r=3.0):
    """Overdispersed counts via a Gamma-Poisson (negative binomial) mixture
    with the same mean path as poisson_counts:

        mu_t as in poisson_counts;  g_t ~ Gamma(r, mu_t/r);
        y_t ~ Poisson(g_t)   =>   Var = mu_t + mu_t^2 / r.

    With r=3, variance is up to ~5x the Poisson variance at the same mean.
    The paired contrast poisson_counts vs negbin_counts isolates
    overdispersion: point forecasts should coincide, predictive intervals
    must not.
    """
    rng = _rng(seed)
    t = _t(T)
    mu = np.exp(1.5 + 0.4 * (t / T) + 0.7 * np.sin(2 * np.pi * t / PERIOD))
    g = rng.gamma(shape=r, scale=mu / r)
    return rng.poisson(g).astype(float)


def skellam_integer(T=DEFAULT_T, seed=None):
    """Signed integers (difference of two seasonal Poisson flows):

        y_t = N1_t - N2_t,  N1 ~ Poi(exp(1.2 + 0.8 s_t)),
                            N2 ~ Poi(exp(1.2 - 0.8 s_t)),
        s_t = sin(2 pi t/24).

    Anti-phase intensities give a seasonal signed count with mean
    2*exp(1.2)*sinh(0.8 s_t) swinging roughly +/-5.9 - integer-valued but
    crossing zero, covering the 'count (negative and positive)' target
    type that pure Poisson tasks cannot.
    """
    rng = _rng(seed)
    s = np.sin(2 * np.pi * _t(T) / PERIOD)
    n1 = rng.poisson(np.exp(1.2 + 0.8 * s))
    n2 = rng.poisson(np.exp(1.2 - 0.8 * s))
    return (n1 - n2).astype(float)


def lognormal_positive(T=DEFAULT_T, seed=None):
    """Strictly positive continuous series, multiplicative on every scale:

        log y_t = 2 + 0.5*(t/T) + 0.4 sin(2 pi t/24) + u_t,
        u_t = AR(1)(phi=0.7, sigma=0.2).

    Level ~7 -> ~12 with conditionally log-normal noise: sd proportional to
    level and right-skewed marginals.  The natural model is additive in
    logs; tests whether a forecaster respects positivity (no negative
    samples/intervals) and multiplicative error structure.
    """
    rng = _rng(seed)
    t = _t(T)
    logy = 2.0 + 0.5 * (t / T) + 0.4 * np.sin(2 * np.pi * t / PERIOD) \
        + _ar1(rng, T, 0.7, 0.2)
    return np.exp(logy)


def intermittent_demand(T=DEFAULT_T, seed=None):
    """Zero-inflated intermittent demand (Croston setting):

        occurrence: o_t ~ Bernoulli(p_t),
                    p_t = sigmoid(-1.2 + 0.8 sin(2 pi t/24))  (~0.12-0.4),
        size:       s_t ~ 1 + Poisson(3),
        y_t = o_t * s_t.

    ~75% zeros with seasonally varying occurrence odds.  Per-step squared
    error is minimised by p_t*E[s] (a fractional value!), not by the modal
    zero - the classic intermittent-demand trap; also probes zero-inflation
    handling in probabilistic outputs.
    """
    rng = _rng(seed)
    t = _t(T)
    p = 1.0 / (1.0 + np.exp(-(-1.2 + 0.8 * np.sin(2 * np.pi * t / PERIOD))))
    occ = rng.random(T) < p
    size = 1 + rng.poisson(3.0, size=T)
    return (occ * size).astype(float)


def intermittent_bursty(T=DEFAULT_T, seed=None, phi=0.95, thresh=0.84):
    """Intermittent demand with serially dependent occurrence:

        z_t = 0.95 z_{t-1} + sqrt(1-0.95^2) eps_t   (latent, stationary sd 1),
        o_t = 1{z_t > 0.84}   (marginal occurrence prob ~0.2, in runs),
        y_t = o_t * (1 + Poisson(2)).

    The persistent latent makes demand arrive in bursts separated by long
    quiet spells - the occurrence indicator has lag-1 autocorrelation
    ~0.5 instead of Croston's i.i.d.-interval assumption.  The optimal
    forecast is state-dependent: P(demand next step) is high right after
    observed demand and decays through a quiet spell.  Models that estimate
    a single average demand rate are systematically wrong in both states.
    """
    rng = _rng(seed)
    sigma = np.sqrt(1.0 - phi**2)
    z = _ar1(rng, T, phi, sigma)
    occ = z > thresh
    size = 1 + rng.poisson(2.0, size=T)
    return (occ * size).astype(float)


def lumpy_demand(T=DEFAULT_T, seed=None, p=0.08, mu=1.0, s=1.0):
    """Lumpy demand (Syntetos-Boylan 'lumpy' quadrant): rare occurrences
    AND highly variable sizes:

        o_t ~ Bernoulli(0.08) i.i.d.  (average inter-demand interval 12.5),
        s_t = ceil(LogNormal(mu=1.0, sigma=1.0))  (heavy-tailed, CV^2 ~ 1.7),
        y_t = o_t * s_t.

    ~92% zeros with sizes ranging from 1 to occasional ~40-unit spikes.
    Complements intermittent_demand (frequent-ish occurrences, low size
    variability): here BOTH the timing and the magnitude are hard, the
    regime where point forecasts are least informative and probabilistic
    (quantile) forecasts carry all the value.  A 512-step context still
    contains ~41 demand events, so the size distribution is estimable.
    """
    rng = _rng(seed)
    occ = rng.random(T) < p
    size = np.ceil(rng.lognormal(mean=mu, sigma=s, size=T))
    return (occ * size).astype(float)


def zero_inflated_continuous(T=DEFAULT_T, seed=None):
    """Zero-inflated *continuous* series (precipitation analogue):

        p_t = sigmoid(-0.8 + 1.2 sin(2 pi t/24))  (occurrence prob ~0.12-0.6),
        a_t ~ LogNormal(0.8, 0.5)                 (positive continuous amount),
        y_t = 1{u_t < p_t} * a_t.

    A mixed discrete-continuous marginal: an atom at exactly 0 plus a
    right-skewed density on (0, inf), with seasonally varying occurrence
    odds.  Unlike the count-valued intermittent tasks, the nonzero part is
    continuous, so quantization tricks do not apply; predictive
    distributions must represent both the zero mass and the continuous
    tail (as in precipitation forecasting).
    """
    rng = _rng(seed)
    t = _t(T)
    p = 1.0 / (1.0 + np.exp(-(-0.8 + 1.2 * np.sin(2 * np.pi * t / PERIOD))))
    occ = rng.random(T) < p
    amount = rng.lognormal(mean=0.8, sigma=0.5, size=T)
    return (occ * amount).astype(float)


# ======================================================================
# 9. Multivariate / cross-series structure
#    (columns documented per function; column 0 is the primary target)
# ======================================================================

def mv_correlated_noise(T=DEFAULT_T, seed=None, m=3, phi=0.7, rho=0.8):
    """m AR(1) series with strongly correlated innovations (returns (T, m)):

        x_t = phi * x_{t-1} + e_t,   e_t ~ N(0, Sigma),
        Sigma_ij = rho^{|i != j|}   (equicorrelation 0.8).

    Because the transition is diagonal, the conditional MEAN of each series
    given the joint past equals its univariate forecast - contemporaneous
    correlation adds nothing to point accuracy.  What it does change is the
    joint predictive distribution.  This task therefore tests joint/
    probabilistic calibration specifically; a gap between a model's
    univariate and multivariate point scores here signals confusion, not
    skill.
    """
    rng = _rng(seed)
    cov = np.full((m, m), rho) + (1.0 - rho) * np.eye(m)
    L = np.linalg.cholesky(cov)
    e = rng.standard_normal((T + BURN_IN, m)) @ L.T
    x = np.zeros((T + BURN_IN, m))
    for t in range(1, T + BURN_IN):
        x[t] = phi * x[t - 1] + e[t]
    return x[BURN_IN:]


def mv_var(T=DEFAULT_T, seed=None, sigma=0.5):
    """Bivariate VAR(1) with genuine cross-dynamics (returns (T, 2)):

        x_t = A x_{t-1} + e_t,  A = [[0.7, 0.25], [-0.2, 0.6]],
        e_t ~ N(0, sigma^2 I).

    Eigenvalues of A are complex with modulus ~0.69 (stationary, damped
    rotational dynamics).  Each series Granger-causes the other, so the
    optimal forecast of either series requires both histories - the
    canonical test that a multivariate model actually uses cross-series
    lags.
    """
    rng = _rng(seed)
    A = np.array([[0.7, 0.25], [-0.2, 0.6]])
    x = np.zeros((T + BURN_IN, 2))
    e = sigma * rng.standard_normal((T + BURN_IN, 2))
    for t in range(1, T + BURN_IN):
        x[t] = A @ x[t - 1] + e[t]
    return x[BURN_IN:]


def mv_leadlag(T=DEFAULT_T, seed=None, lag=8, beta=1.0, sigma_y=0.3,
               phi=0.95, sigma_x=1.0):
    """Leading indicator: x leads y by ``lag`` steps (returns (T, 2) =
    [y, x]):

        x_t = 0.95 x_{t-1} + eps_t,      y_t = beta * x_{t-lag} + 0.3 nu_t.

    For horizons h <= lag, y_{t+h} is (up to small noise) *already visible*
    in x's context - the multivariate Bayes MSE is 0.09 while the best
    univariate forecast inherits x's innovation variance.  The purest test
    that a model exploits cross-series lead-lag information; with h > lag
    the advantage decays, so keep lag >= h/2 relative to the task horizon.
    """
    rng = _rng(seed)
    x_full = _ar1(rng, T + lag, phi, sigma_x)
    x = x_full[lag:]
    y = beta * x_full[:T] + sigma_y * rng.standard_normal(T)
    return np.column_stack([y, x])


def mv_common_factor(T=DEFAULT_T, seed=None, m=4):
    """Factor structure: m=4 series driven by one latent AR(1) factor
    (returns (T, 4)):

        f_t = 0.9 f_{t-1} + eps_t;   y_it = loading_i * f_t + u_it,
        loadings = (1.0, 0.8, -0.6, 0.5);  u_it = AR(1)(0.4, 1.0) idiosync.

    The common factor carries ~70% of total panel variance - dominant but
    not trivially so.  Cross-sectional averaging de-noises the factor: the
    panel reveals f_t more precisely than any single series does, so pooled
    forecasts strictly beat per-series univariate ones.  Tests factor
    extraction / cross-sectional information pooling.
    """
    rng = _rng(seed)
    loadings = np.array([1.0, 0.8, -0.6, 0.5])
    f = _ar1(rng, T, 0.9, 1.0)
    out = np.empty((T, m))
    for i in range(m):
        out[:, i] = loadings[i] * f + _ar1(rng, T, 0.4, 1.0)
    return out


def mv_cointegrated(T=DEFAULT_T, seed=None):
    """Cointegrated pair sharing one stochastic trend (returns (T, 2)):

        w_t = w_{t-1} + eps_t (random walk);
        y1_t = w_t + u1_t,   y2_t = 0.7 w_t + u2_t,
        u_i = AR(1)(0.5, 0.5) stationary.

    Each series alone is a unit-root process, but the spread y1 - y2/0.7
    is stationary: deviations between the series are temporary and
    forecastably close.  Tests whether a multivariate model exploits the
    error-correction structure (forecasting the *pair* coherently) instead
    of forecasting two independent random walks.
    """
    rng = _rng(seed)
    w = np.cumsum(rng.standard_normal(T))
    y1 = w + _ar1(rng, T, 0.5, 0.5)
    y2 = 0.7 * w + _ar1(rng, T, 0.5, 0.5)
    return np.column_stack([y1, y2])


def covariate_nonlinear(T=DEFAULT_T, seed=None, lag=2, sigma_y=0.3):
    """Covariate-driven target with a nonlinear, lagged link (returns
    (T, 2) = [y, x]; x is the covariate, known over context+horizon per the
    TempusBench task definition):

        x_t = 5 sin(2 pi t/24) + AR(1)(0.8, 1.0),
        y_t = 4 tanh(x_{t-lag} / 3) + sigma_y * nu_t.

    Given the covariate path, y is almost deterministic - but only through
    a saturating nonlinearity and a 2-step lag.  Linear covariate handling
    leaves large errors at |x| extremes; ignoring the covariate leaves the
    task nearly unpredictable at long horizons.
    """
    rng = _rng(seed)
    t_full = np.arange(-lag, T, dtype=float)
    x_full = 5.0 * np.sin(2 * np.pi * t_full / PERIOD) \
        + _ar1(rng, T + lag, 0.8, 1.0)
    x = x_full[lag:]
    y = 4.0 * np.tanh(x_full[:T] / 3.0) + sigma_y * rng.standard_normal(T)
    return np.column_stack([y, x])


# ======================================================================
# Registry
# ======================================================================

# Category vocabulary (a task may carry several tags):
#   trend, stationarity, seasonality, noise, memory, nonlinearity,
#   structural_change, target_type, intermittency, multivariate,
#   covariate
TASKS = {
    # --- baseline / controls ----------------------------------------
    "iid_gaussian": dict(fn=iid_gaussian, variate="univariate",
                         target_type="continuous_real",
                         categories=["noise", "stationarity"]),
    "noise_free_composite": dict(fn=noise_free_composite, variate="univariate",
                                 target_type="continuous_real",
                                 categories=["seasonality", "trend"]),
    "low_snr_seasonal": dict(fn=low_snr_seasonal, variate="univariate",
                             target_type="continuous_real",
                             categories=["seasonality", "noise"]),
    "ma1": dict(fn=ma1, variate="univariate",
                target_type="continuous_real",
                categories=["memory", "stationarity"]),
    # --- trend / movement -------------------------------------------
    "mean_reverting_ar1": dict(fn=mean_reverting_ar1, variate="univariate",
                               target_type="continuous_real",
                               categories=["stationarity", "memory"]),
    "near_unit_root_ar1": dict(fn=near_unit_root_ar1, variate="univariate",
                               target_type="continuous_real",
                               categories=["stationarity", "memory",
                                           "trend"]),
    "random_walk": dict(fn=random_walk, variate="univariate",
                        target_type="continuous_real",
                        categories=["trend", "memory"]),
    "random_walk_drift": dict(fn=random_walk_drift, variate="univariate",
                              target_type="continuous_real",
                              categories=["trend", "memory"]),
    "linear_trend": dict(fn=linear_trend, variate="univariate",
                         target_type="continuous_real",
                         categories=["trend"]),
    "exponential_trend": dict(fn=exponential_trend, variate="univariate",
                              target_type="continuous_positive",
                              categories=["trend"]),
    "log_trend": dict(fn=log_trend, variate="univariate",
                      target_type="continuous_positive",
                      categories=["trend"]),
    "power_trend": dict(fn=power_trend, variate="univariate",
                        target_type="continuous_positive",
                        categories=["trend"]),
    "logistic_trend": dict(fn=logistic_trend, variate="univariate",
                           target_type="continuous_positive",
                           categories=["trend", "nonlinearity"]),
    "piecewise_linear_trend": dict(fn=piecewise_linear_trend,
                                   variate="univariate",
                                   target_type="continuous_real",
                                   categories=["trend", "structural_change"]),
    # --- seasonality -------------------------------------------------
    "sinusoidal_seasonal": dict(fn=sinusoidal_seasonal, variate="univariate",
                                target_type="continuous_real",
                                categories=["seasonality", "stationarity",
                                            "noise"]),
    "multi_seasonal": dict(fn=multi_seasonal, variate="univariate",
                           target_type="continuous_real",
                           categories=["seasonality"]),
    "nonsinusoidal_seasonal": dict(fn=nonsinusoidal_seasonal,
                                   variate="univariate",
                                   target_type="continuous_real",
                                   categories=["seasonality"]),
    "trend_seasonal_additive": dict(fn=trend_seasonal_additive,
                                    variate="univariate",
                                    target_type="continuous_positive",
                                    categories=["seasonality", "trend",
                                                "noise"]),
    "trend_seasonal_multiplicative": dict(fn=trend_seasonal_multiplicative,
                                          variate="univariate",
                                          target_type="continuous_positive",
                                          categories=["seasonality", "trend",
                                                      "noise"]),
    "damped_seasonal": dict(fn=damped_seasonal, variate="univariate",
                            target_type="continuous_real",
                            categories=["seasonality", "trend"]),
    "irregular_period_seasonal": dict(fn=irregular_period_seasonal,
                                      variate="univariate",
                                      target_type="continuous_real",
                                      categories=["seasonality"]),
    "evolving_seasonal": dict(fn=evolving_seasonal, variate="univariate",
                              target_type="continuous_real",
                              categories=["seasonality",
                                          "structural_change"]),
    # --- noise -------------------------------------------------------
    "heteroskedastic_level": dict(fn=heteroskedastic_level,
                                  variate="univariate",
                                  target_type="continuous_positive",
                                  categories=["noise"]),
    "heteroskedastic_time": dict(fn=heteroskedastic_time,
                                 variate="univariate",
                                 target_type="continuous_real",
                                 categories=["noise", "structural_change"]),
    "garch_noise": dict(fn=garch_noise, variate="univariate",
                        target_type="continuous_real",
                        categories=["noise", "memory"]),
    "heavy_tailed_noise": dict(fn=heavy_tailed_noise, variate="univariate",
                               target_type="continuous_real",
                               categories=["noise"]),
    "skewed_noise": dict(fn=skewed_noise, variate="univariate",
                         target_type="continuous_real",
                         categories=["noise"]),
    "autocorrelated_noise": dict(fn=autocorrelated_noise,
                                 variate="univariate",
                                 target_type="continuous_real",
                                 categories=["noise", "memory"]),
    "outlier_contaminated": dict(fn=outlier_contaminated,
                                 variate="univariate",
                                 target_type="continuous_real",
                                 categories=["noise"]),
    "measurement_error_rw": dict(fn=measurement_error_rw,
                                 variate="univariate",
                                 target_type="continuous_real",
                                 categories=["noise", "memory", "trend"]),
    # --- memory ------------------------------------------------------
    "ar2_pseudocyclic": dict(fn=ar2_pseudocyclic, variate="univariate",
                             target_type="continuous_real",
                             categories=["memory", "seasonality",
                                         "stationarity"]),
    "long_memory_fgn": dict(fn=long_memory_fgn, variate="univariate",
                            target_type="continuous_real",
                            categories=["memory", "stationarity"]),
    # --- nonlinearity ------------------------------------------------
    "setar": dict(fn=setar, variate="univariate",
                  target_type="continuous_real",
                  categories=["nonlinearity", "memory",
                              "structural_change"]),
    "logistic_map": dict(fn=logistic_map, variate="univariate",
                         target_type="continuous_positive",
                         categories=["nonlinearity"]),
    "mackey_glass": dict(fn=mackey_glass, variate="univariate",
                         target_type="continuous_positive",
                         categories=["nonlinearity", "seasonality"]),
    # --- structural change -------------------------------------------
    "level_shift": dict(fn=level_shift, variate="univariate",
                        target_type="continuous_real",
                        categories=["structural_change", "seasonality"]),
    "variance_shift": dict(fn=variance_shift, variate="univariate",
                           target_type="continuous_real",
                           categories=["structural_change", "noise"]),
    "markov_switching_ar": dict(fn=markov_switching_ar, variate="univariate",
                                target_type="continuous_real",
                                categories=["structural_change", "memory",
                                            "nonlinearity"]),
    # --- target type -------------------------------------------------
    "binary_latent_ar": dict(fn=binary_latent_ar, variate="univariate",
                             target_type="binary",
                             categories=["target_type", "seasonality",
                                         "memory"]),
    "ordinal_categorical": dict(fn=ordinal_categorical, variate="univariate",
                                target_type="categorical_ordinal",
                                categories=["target_type", "seasonality",
                                            "memory"]),
    "poisson_counts": dict(fn=poisson_counts, variate="univariate",
                           target_type="count_positive",
                           categories=["target_type", "seasonality",
                                       "trend"]),
    "negbin_counts": dict(fn=negbin_counts, variate="univariate",
                          target_type="count_positive",
                          categories=["target_type", "noise"]),
    "skellam_integer": dict(fn=skellam_integer, variate="univariate",
                            target_type="count_signed",
                            categories=["target_type", "seasonality"]),
    "lognormal_positive": dict(fn=lognormal_positive, variate="univariate",
                               target_type="continuous_positive",
                               categories=["target_type", "noise",
                                           "seasonality"]),
    "intermittent_demand": dict(fn=intermittent_demand, variate="univariate",
                                target_type="count_positive",
                                categories=["target_type", "intermittency",
                                            "seasonality"]),
    "intermittent_bursty": dict(fn=intermittent_bursty, variate="univariate",
                                target_type="count_positive",
                                categories=["intermittency", "target_type",
                                            "memory"]),
    "lumpy_demand": dict(fn=lumpy_demand, variate="univariate",
                         target_type="count_positive",
                         categories=["intermittency", "target_type",
                                     "noise"]),
    "zero_inflated_continuous": dict(fn=zero_inflated_continuous,
                                     variate="univariate",
                                     target_type="continuous_zero_inflated",
                                     categories=["intermittency",
                                                 "target_type",
                                                 "seasonality"]),
    # --- multivariate / covariate -----------------------------------
    "mv_correlated_noise": dict(fn=mv_correlated_noise,
                                variate="multivariate",
                                target_type="continuous_real",
                                categories=["multivariate", "noise"]),
    "mv_var": dict(fn=mv_var, variate="multivariate",
                   target_type="continuous_real",
                   categories=["multivariate", "memory"]),
    "mv_leadlag": dict(fn=mv_leadlag, variate="multivariate",
                       target_type="continuous_real",
                       categories=["multivariate", "covariate", "memory"]),
    "mv_common_factor": dict(fn=mv_common_factor, variate="multivariate",
                             target_type="continuous_real",
                             categories=["multivariate", "memory"]),
    "mv_cointegrated": dict(fn=mv_cointegrated, variate="multivariate",
                            target_type="continuous_real",
                            categories=["multivariate", "trend", "memory"]),
    "covariate_nonlinear": dict(fn=covariate_nonlinear, variate="covariate",
                                target_type="continuous_real",
                                categories=["covariate", "nonlinearity",
                                            "seasonality"]),
}


def generate(name, T=DEFAULT_T, seed=None, **kwargs):
    """Generate the dataset for a registered task by name."""
    return TASKS[name]["fn"](T=T, seed=seed, **kwargs)


def generate_all(T=DEFAULT_T, seed=0):
    """Generate every task dataset (seed offset per task for independence).

    Returns {task_name: ndarray}.  Task order (and therefore the per-task
    seed) is the fixed registry order, so results are reproducible.
    """
    return {name: spec["fn"](T=T, seed=seed + i)
            for i, (name, spec) in enumerate(TASKS.items())}
