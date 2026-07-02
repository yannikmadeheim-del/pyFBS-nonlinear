import numpy as np

from pyfbs.nonlinearFBS import NumericalFRF, ModalVPFRF


def _setup(seed, n_modes, n_chn, n_imp, N_vp, zeta=0.02):
    rng = np.random.default_rng(seed)
    Omega = np.sort(rng.uniform(50.0, 500.0, n_modes))
    Phi_c = rng.standard_normal((n_chn, n_modes))     # physical channel modes
    Phi_i = rng.standard_normal((n_imp, n_modes))     # physical impact  modes
    Tu    = rng.standard_normal((N_vp, n_chn))        # VPT response  matrix
    Tf    = rng.standard_normal((n_imp, N_vp))        # VPT load      matrix
    # physical provider over stacked [channels ; impacts]
    phys  = NumericalFRF.from_modal(Omega, np.vstack([Phi_c, Phi_i]), modal_damping=zeta)
    chn, imp = np.arange(n_chn), n_chn + np.arange(n_imp)
    # folded VP provider
    fold  = ModalVPFRF(Omega, Tu @ Phi_c, Tf.T @ Phi_i, modal_damping=zeta)
    return phys, fold, Tu, Tf, chn, imp, N_vp


def test_fold_matches_runtime_vpt():
    """ModalVPFRF(Psi_c, Psi_i) == Tu @ Y_phys @ Tf  (== VPT.apply_vpt: tu @ frf @ tf)."""
    phys, fold, Tu, Tf, chn, imp, N_vp = _setup(0, 8, 6, 5, 3)
    harmonics, omega = np.array([0, 1, 2, 3]), 123.4
    vp = np.arange(N_vp)

    Y_phys = phys.compute_FRF(omega, harmonics, chn, imp)        # (Nh, n_chn, n_imp)
    Y_ref  = Tu @ Y_phys @ Tf                                    # apply_vpt's operation
    Y_fold = fold.compute_FRF(omega, harmonics, vp, vp)          # synthesized directly in VP space

    assert Y_fold.shape == (len(harmonics), N_vp, N_vp)
    assert np.allclose(Y_fold, Y_ref, rtol=1e-10, atol=1e-12)


def test_fold_derivative_matches_runtime_vpt():
    """d/dw of the fold == Tu @ dY_phys/dw @ Tf (Tu/Tf constant in omega)."""
    phys, fold, Tu, Tf, chn, imp, N_vp = _setup(1, 6, 4, 4, 2)
    harmonics, omega = np.array([1, 2]), 80.0
    vp = np.arange(N_vp)

    dY_phys = phys.compute_FRF_derivative(omega, harmonics, chn, imp, None)
    dY_ref  = Tu @ dY_phys @ Tf
    dY_fold = fold.compute_FRF_derivative(omega, harmonics, vp, vp, None)
    assert np.allclose(dY_fold, dY_ref, rtol=1e-10, atol=1e-12)


def test_n_dofs_is_Nvp():
    _, fold, _, _, _, _, N_vp = _setup(2, 5, 4, 3, 2)
    assert fold.n_dofs == N_vp        # == B_coupling.shape[1]; plain FBSProblem consumes it
