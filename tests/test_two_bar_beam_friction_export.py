"""The two_bar_beam_friction result CSV carries the forces the solver converged on.

The force block is what the contact-force figure plots, and for the DLFT runs it
is the ONLY record of the normal force -- rigid contact makes it a
prediction/correction quantity, not a function of the exported displacements. So
the checks here are on the export contract, per method:

  * AFT:  the exported normal force is the softplus law evaluated on the
          exported normal displacement;
  * both: the exported friction force is mu * N(t) * tanh(alpha1 * d/dt x_T),
          with N(t) the exported normal force.

Both identities are checked on the run's OWN time grid (n_t = sample_number) and
projected back onto the retained harmonics, which is exactly what the solver
does -- so they hold to round-off, not just to truncation error.

Run with:  python -m pytest tests/test_two_bar_beam_friction_export.py --noconftest
"""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from numpy.fft import rfft

from pyfbs.nonlinearFBS.examples.two_bar_beam_friction import main as run
from pyfbs.nonlinearFBS.examples.two_bar_beam_friction import plotting_saving as io
from pyfbs.nonlinearFBS.examples.two_bar_beam_friction.dynamical_system import (
    _softplus_normal, BarBeamParams)

HARMONICS = list(range(0, 8))
SAMPLE_NUMBER = 64            # >= 4H+1 = 29


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """A short branch per method, written and read back through the real IO."""
    out = {}
    directory = tmp_path_factory.mktemp("two_bar_forces")
    for method in ("aft", "dlft_aft"):
        cfg = deepcopy(run.CONFIG)
        cfg.update(method=method, harmonics=HARMONICS, sample_number=SAMPLE_NUMBER,
                   omega_lo=0.95, omega_hi=0.97, maximum_number_of_solutions=6)
        provider = run.build_provider(cfg, "modal")
        system, problem, ss, solve_time = run.solve_config(cfg, provider)
        path = directory / f"{method}.csv"
        io.save_solution(path, cfg, ss, problem, system, method, "modal", solve_time)
        out[method] = (path, cfg)
    return out


def _amplitudes(df, label, harmonics):
    """(n, Nh) complex physical amplitudes of one exported channel."""
    return np.column_stack([df[f"re_h{h}_{label}"].to_numpy()
                            + 1j * df[f"im_h{h}_{label}"].to_numpy()
                            for h in harmonics])


def _retained(signal, harmonics, n_t):
    """rFFT of a time signal, keeping the harmonics the solver retains."""
    return rfft(signal)[harmonics]


def _bins(amplitudes, harmonics, n_t):
    """Physical amplitudes back onto rFFT bins, the solver's own convention."""
    scale = np.array([(1.0 if h == 0 else 2.0) / n_t for h in harmonics])
    return np.asarray(amplitudes) / scale


@pytest.mark.parametrize("method", ["aft", "dlft_aft"])
def test_force_block_is_read_back(solved, method):
    path, cfg = solved[method]
    curve = io.read_result(path)

    assert curve["force_labels"] == list(io.FORCE_LABELS)
    assert curve["force_amp"] is not None
    assert curve["force_amp"].shape == (len(curve["omega"]), len(HARMONICS), 2)
    assert np.all(np.isfinite(curve["force_amp"]))
    # the preload is static, so the normal force MUST have a harmonic-0 part
    assert np.all(np.abs(curve["force_amp"][:, HARMONICS.index(0), 0]) > 0.0)


def test_aft_normal_force_is_the_softplus_law(solved):
    """AFT exports N = softplus(k*alpha2*(x_N - eps))/alpha2 of the exported x_N."""
    path, cfg = solved["aft"]
    p = BarBeamParams(**cfg["params"])
    df = pd.read_csv(path, comment="#")
    n_t = cfg["sample_number"]

    xN = _amplitudes(df, "x_rel_N", HARMONICS)
    fN = _amplitudes(df, "f_N", HARMONICS)
    for row in range(len(df)):
        _, xN_t = io.period_signal(xN[row], HARMONICS, n_t=n_t)
        law, _ = _softplus_normal(xN_t, p)
        assert np.allclose(_retained(law, HARMONICS, n_t),
                           _bins(fN[row], HARMONICS, n_t), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method", ["aft", "dlft_aft"])
def test_friction_force_follows_the_normal_force(solved, method):
    """f_T = mu * N(t) * tanh(alpha1 * d/dt x_T) -- the same law in both methods.

    WHICH N(t) is exact differs, and that is a real property of the two schemes,
    not a detail of the export:

      * DLFT drives the friction law with ``x.f_normal``, the harmonic
        TRUNCATION of the contact force -- exactly what lands in the CSV. The
        identity therefore closes on the exported f_N to round-off.
      * AFT evaluates N pointwise from x_N and truncates only the PRODUCT
        mu*N*tanh(...). Once N carries content beyond the retained harmonics,
        the exported (truncated) N no longer reproduces the product exactly, so
        the identity has to be checked against the pointwise law instead.
    """
    path, cfg = solved[method]
    p = BarBeamParams(**cfg["params"])
    df = pd.read_csv(path, comment="#")
    n_t = cfg["sample_number"]

    xN = _amplitudes(df, "x_rel_N", HARMONICS)
    xT = _amplitudes(df, "x_rel_T", HARMONICS)
    fN = _amplitudes(df, "f_N", HARMONICS)
    fT = _amplitudes(df, "f_T", HARMONICS)
    omega = df["omega_rad_s"].to_numpy()
    for row in range(len(df)):
        # velocity amplitudes: udot(t) = Re( sum_h 1j h omega a_h e^{1j h tau} )
        vT = 1j * np.array(HARMONICS) * omega[row] * xT[row]
        _, vT_t = io.period_signal(vT, HARMONICS, n_t=n_t)
        if method == "aft":
            _, xN_t = io.period_signal(xN[row], HARMONICS, n_t=n_t)
            N_t, _ = _softplus_normal(xN_t, p)
        else:
            _, N_t = io.period_signal(fN[row], HARMONICS, n_t=n_t)
        law = p.mu * N_t * np.tanh(p.alpha1 * vT_t)
        assert np.allclose(_retained(law, HARMONICS, n_t),
                           _bins(fT[row], HARMONICS, n_t), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method", ["aft", "dlft_aft"])
def test_normal_force_is_compressive(solved, method):
    """A contact force cannot pull, beyond what the truncation puts there.

    The mean is strictly positive; the pointwise signal is allowed a Gibbs
    undershoot, which is large here because the contact opens and closes and
    ``HARMONICS`` only resolves the resulting pulse with 8 terms. The bound
    scales with the retained harmonics, so it is deliberately generous rather
    than tuned to the current run.
    """
    path, cfg = solved[method]
    df = pd.read_csv(path, comment="#")
    fN = _amplitudes(df, "f_N", HARMONICS)
    for row in range(len(df)):
        _, N_t = io.period_signal(fN[row], HARMONICS)
        assert N_t.mean() > 0.0
        assert N_t.min() > -0.3 * N_t.max()
