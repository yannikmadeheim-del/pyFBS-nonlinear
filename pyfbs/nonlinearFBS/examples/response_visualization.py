"""Full-field animation of a converged nonlinear FBS forced response.

Mode shapes animate a single real vector as ``Re(phi e^{i w t})``.  A nonlinear
HBM solution at a given excitation frequency is richer: every retained harmonic
contributes, so the true motion is the full periodic time series
``u(t) = sum_n Re(U_n e^{i n w t})``.  This module reconstructs that motion on the
FULL FE mesh of each substructure and feeds one period of frames to the SAME
mode-shape animator (``View3D._add_modeshape``), so the result looks exactly like a
mode-shape animation but shows the actual operating motion at the chosen frequency.

How the full-mesh field is recovered
-------------------------------------
The solver works in a small reduced space (reference channels + 6 virtual-point
DoFs per substructure) and the response is computed as ``u = Y (F_ext - B^T F_int)``
restricted to the input DoFs.  We keep that exact load
``L_in = F_ext - B^T F_int`` from the converged point and re-synthesize the OUTPUT
side onto the mesh: replace the VP-projected output modes used by
:class:`ModalVPFRF` with the substructure's FULL mesh modes ``model.eig_vec`` while
keeping the same VP-projected INPUT modes ``Psi_i = Tf^T Phi_i`` and the same modal
poles/damping.  This is the identical modal sum, so the mesh field stays consistent
with the reduced solution the solver actually converged.

The excitation and output DoFs are marked in the scene as red (impact-style) and
blue (channel-style) arrows.  Because the re-synthesis is modal, a solve driven by an
:class:`ExperimentalFRF` is animated as the modal reconstruction consistent with the
converged interface load (a splined admittance grid carries no mesh information).
"""

import numpy as np
import pandas as pd

from ..frequency_domain import Fourier, Fourier_Real, FourierOmegaPoint
from ...mck.mck import Model


def _full_mesh_response_coeffs(problem, fourier, omega, substructures, modal_damping):
    """Per-harmonic full-mesh displacement coefficients for each substructure.

    :param problem: the :class:`FBSProblem` that produced the solution (gives the
        interface load and the input-DoF bookkeeping the solver used).
    :param fourier: converged interface ``Fourier`` coefficients (the Newton unknown).
    :param omega: physical excitation frequency [rad/s] of this branch point.
    :param substructures: list of dicts in the SAME block order as the global reduced
        DoF vector, each ``{"model", "vpt", "df_imp", "n_reduced"}``:
        ``model`` a pyFBS Model with a real modal solution, ``vpt`` its VPT,
        ``df_imp`` the location-updated impact dataframe, ``n_reduced`` the number of
        reduced DoFs this substructure occupies in the global vector.
    :param modal_damping: modal damping ratio (scalar or per-mode) used to build the
        FRF provider -- must match so the mesh field matches the reduced solution.
    :returns: list of ``(model, U)`` with ``U`` the ``(Nh, 3*n_nodes, 1)`` complex
        per-harmonic mesh displacement of that substructure.
    """
    x = FourierOmegaPoint(fourier, omega)
    Nh = Fourier.number_of_harmonics

    # Interface force harmonics from the converged gap -- the same call the solver's
    # own post-processing (compute_full_response) uses, so signs/scaling match.
    Fnl = problem.method.compute_F_int(x, problem.ode).reshape(Nh, problem.d_int, 1)
    L_in = problem.F_ext - problem.B.T @ Fnl          # (Nh, n_in, 1): load on in_dofs
    in_dofs = problem.in_dofs
    omegas = Fourier.harmonics.astype(float) * omega  # query the exact n*omega

    out = []
    offset = 0
    for sub in substructures:
        model, vpt = sub["model"], sub["vpt"]
        df_imp, n_red = sub["df_imp"], sub["n_reduced"]

        # VP-projected INPUT modes -- identical to ModalVPFRF.from_substructures.
        eigval2, damp, Phi_i = model.transform_modal_parameters(
            df_imp, modal_damping=modal_damping, return_channel_only=True)
        Psi_i = vpt.tf.T @ Phi_i                       # (n_red, n_modes)
        Omega = np.sqrt(eigval2)                       # poles unchanged by the VPT
        Phi_mesh = model.eig_vec[:, :len(Omega)]       # (3*n_nodes, n_modes) FULL mesh

        # Which global input DoFs fall in this substructure's reduced block (the rest
        # are zeroed by the block-diagonal structure, so we just skip them).
        mask = (in_dofs >= offset) & (in_dofs < offset + n_red)
        if mask.any():
            local_in = in_dofs[mask] - offset
            _, Y_mesh = Model.custom_frf_synth(
                Omega, Phi_mesh, Psi_i[local_in],
                modal_damping=damp, omegas=omegas)     # (Nh, 3*n_nodes, |in_s|)
            U = np.einsum('nij,njk->nik', Y_mesh, L_in[:, mask, :])  # (Nh, 3*n_nodes, 1)
        else:
            U = np.zeros((Nh, Phi_mesh.shape[0], 1), dtype=complex)

        out.append((model, U))
        offset += n_red
    return out


def _coeffs_to_node_frames(U):
    """``(Nh, 3*n_nodes, 1)`` complex harmonics -> ``(n_nodes, 3, Nt)`` real frames.

    The inverse real FFT reconstructs one full period (Nt = sample_number samples);
    ``eig_vec`` rows are node-major with x,y,z innermost, so the reshape aligns each
    triplet with a mesh node.
    """
    f = Fourier(U)
    Fourier_Real.compute_time_series(f)                # (Nt, 3*n_nodes, 1)
    ts = f.time_series[:, :, 0]                        # (Nt, 3*n_nodes), real
    Nt = ts.shape[0]
    ts = ts.reshape(Nt, -1, 3)                         # (Nt, n_nodes, 3)
    return np.moveaxis(ts, 0, 2)                       # (n_nodes, 3, Nt)


def _dof_to_substructure(substructures, dof):
    """Map a global reduced DoF index to ``(substructure dict, local row index)``."""
    offset = 0
    for sub in substructures:
        if dof < offset + sub["n_reduced"]:
            return sub, dof - offset
        offset += sub["n_reduced"]
    raise IndexError(f"reduced DoF {dof} outside the {offset} stacked DoFs")


def _show_io_arrows(view, problem, substructures, output_dof, size):
    """Mark the excitation and output DoFs in the 3D scene.

    Red impact-style arrows: position/direction of every excited input DoF (the
    nonzero harmonics of ``problem.F_ext``).  Blue channel-style arrow: position/
    direction of the plotted output DoF.  Rows are taken from the VPT-transformed
    dataframes ``vpt.df_imp`` / ``vpt.df_chn``, whose row order matches the reduced
    FRF DoFs; rows without direction data (virtual-point DoFs) are skipped.
    """
    dir_cols = ["Direction_1", "Direction_2", "Direction_3"]
    excited = problem.in_dofs[np.any(problem.F_ext[:, :, 0] != 0.0, axis=0)]
    if excited.size:
        rows = [_dof_to_substructure(substructures, int(d)) for d in excited]
        df_exc = pd.concat([sub["vpt"].df_imp.iloc[[k]] for sub, k in rows])
        df_exc = df_exc[df_exc[dir_cols].notna().all(axis=1)]
        if len(df_exc):
            view.show_imp(df_exc, size=size)
    if output_dof is not None:
        sub, k = _dof_to_substructure(substructures, int(output_dof))
        df_out = sub["vpt"].df_chn.iloc[[k]]
        if df_out[dir_cols].notna().all(axis=None):
            view.show_chn(df_out, size=size)


def animate_response_at_frequency(view, problem, solution_set, substructures,
                                  target_frequency, modal_damping,
                                  output_dof=None, show_io=True, io_arrow_size=None,
                                  r_scale=0.08, no_frames=60, no_of_repetitions=2,
                                  fps=30, run_animation=True,
                                  cmap="turbo", show_edges=False, opacity=1.0,
                                  mesh_name="nonlinear_response", mesh_kwargs=None):
    """Animate the full nonlinear periodic motion of the coupled substructures at the
    converged branch point nearest ``target_frequency`` -- exactly like the mode-shape
    animator, but reconstructed from ALL retained harmonics on the full FE mesh.

    :param view: a :class:`pyfbs.display.View3D` (the caller owns the window).
    :param problem: the :class:`FBSProblem` whose solution is ``solution_set``.
    :param solution_set: the :class:`SolutionSet` from ``solve_and_continue``.
    :param substructures: list of ``{"model", "vpt", "df_imp", "n_reduced"}`` in the
        global reduced-DoF block order (see :func:`_full_mesh_response_coeffs`).
    :param target_frequency: desired excitation frequency [Hz]; the nearest converged
        branch point is animated.
    :param modal_damping: modal damping ratio used to build the FRF provider.
    :param output_dof: global reduced DoF index of the plotted response channel,
        marked as a blue channel arrow (None = no output arrow).
    :param show_io: mark the excitation (red) and output (blue) positions/directions.
    :param io_arrow_size: arrow length in model units; None = 0.08 x the merged
        bounding-box diagonal.
    :param r_scale: largest animated nodal displacement as a fraction of the model
        bounding-box diagonal (visual scaling only; relative motion is preserved).
    :param no_frames: frames sampled across one period.
    :param no_of_repetitions: how many periods to play per animation run.
    :param fps: animation frame rate.
    :param run_animation: play once immediately (the toolbar button replays it).
    :returns: ``(index, frequency_Hz)`` of the animated branch point.
    """
    f_branch = np.asarray(solution_set.omega) / (2.0 * np.pi)
    i = int(np.argmin(np.abs(f_branch - target_frequency)))
    fourier, omega = solution_set.fourier[i], solution_set.omega[i]
    print(f"Animating full-mesh response at {f_branch[i]:.2f} Hz "
          f"(nearest to target {target_frequency:.2f} Hz; branch point {i} of "
          f"{len(f_branch)}).")

    coeffs = _full_mesh_response_coeffs(problem, fourier, omega, substructures,
                                        modal_damping)

    # Merge every substructure's FULL mesh into one body (merge_points=False keeps the
    # point order [mesh_0; mesh_1; ...]) and stack the matching per-node frames, so a
    # single mode-shape animation deforms all substructures together.
    combined = coeffs[0][0].mesh.copy()
    node_frames = [_coeffs_to_node_frames(coeffs[0][1])]
    for model, U in coeffs[1:]:
        combined = combined.merge(model.mesh, merge_points=False, inplace=False)
        node_frames.append(_coeffs_to_node_frames(U))
    frames = np.concatenate(node_frames, axis=0)       # (n_pts_total, 3, Nt)

    # One period -> requested frame count, then tile for a few repetitions.
    Nt = frames.shape[2]
    idx = np.linspace(0, Nt, no_frames, endpoint=False).astype(int)
    frames = np.tile(frames[:, :, idx], (1, 1, max(1, int(no_of_repetitions))))

    # Visual scaling: largest nodal motion -> r_scale * model diagonal.
    b = np.asarray(combined.bounds)
    diag = float(np.linalg.norm(b[1::2] - b[0::2]))
    peak = float(np.max(np.abs(frames)))
    if peak > 0.0:
        frames = frames * (r_scale * diag / peak)

    add_kwargs = dict(scalars=np.ones(combined.points.shape[0]), cmap=cmap,
                      show_edges=show_edges, opacity=opacity)
    if mesh_kwargs:
        add_kwargs.update(mesh_kwargs)
    view.plot.add_mesh(combined, name=mesh_name, **add_kwargs)

    if show_io:
        _show_io_arrows(view, problem, substructures, output_dof,
                        io_arrow_size if io_arrow_size is not None else 0.08 * diag)

    mode_dict = {
        "animation_pts": frames,
        "animation_pts_secondary": None,
        "animate_secondary_mode_shape": False,
        "fps": fps,
        "or_pts": combined.points.copy(),
        "mesh": combined,
        "scalars": True,
    }
    view._add_modeshape(mode_dict, run_animation=run_animation, add_note=False)
    return i, float(f_branch[i])
