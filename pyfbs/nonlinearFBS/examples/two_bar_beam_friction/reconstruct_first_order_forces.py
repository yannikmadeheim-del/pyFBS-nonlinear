"""Add the interface force block to a first-order result CSV.

The first-order solver exports the kinematics only. Its force laws are fully
stated in the CSV header, so the converged interface force follows from the
stored harmonics:

    N   = softplus(k*alpha2*(x_N - eps)) / alpha2
    f_T = mu * N * tanh(alpha1 * d/dt x_T)

sampled on the solver's own AFT grid (N_t from the header) and projected back
onto the retained harmonics -- the same truncated force the solver balanced.
The result gets the f_N, f_T columns in the layout of plotting_saving.save_solution,
so read_result picks them up like a native force block.

Run with::

    python reconstruct_first_order_forces.py results/no_sep_runs/first_order.csv
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.fft import irfft, rfft

from pyfbs.nonlinearFBS.examples.two_bar_beam_friction import plotting_saving as io

HERE = Path(__file__).resolve().parent
LAW_PARAMS = ("k", "alpha2", "eps", "mu", "alpha1")


def header_values(header):
    """N_t and the force-law parameters, as stated in the header."""
    text = "\n".join(header)
    n_t = re.search(r"N_t = (\d+)", text)
    if n_t is None:
        raise ValueError("header states no AFT sample count 'N_t = ...'")
    params = {}
    for name in LAW_PARAMS:
        match = re.search(rf"\b{name}=([-+.\deE]+)", text)
        if match is None:
            raise ValueError(f"header states no value for {name}")
        params[name] = float(match.group(1))
    return int(n_t.group(1)), params


def interface_force(df, harmonics, n_t, p):
    """(n, Nh) physical amplitudes of N and f_T at every branch point."""
    h = np.array(harmonics)
    scale = np.where(h == 0, 1.0, 2.0) / n_t

    def amplitudes(label):
        return np.column_stack([df[f"re_h{k}_{label}"] + 1j * df[f"im_h{k}_{label}"]
                                for k in harmonics])

    def time_series(a):
        padded = np.zeros((len(df), n_t // 2 + 1), dtype=complex)
        padded[:, harmonics] = a / scale
        return irfft(padded, n=n_t, axis=1)

    omega = df["omega_rad_s"].to_numpy()[:, None]
    x_N = time_series(amplitudes("x_rel_N"))
    v_T = time_series(1j * h * amplitudes("x_rel_T")) * omega

    c = p["k"] * p["alpha2"] * (x_N - p["eps"])
    # stable softplus: log1p(exp(-|c|)) + max(c, 0)
    N = (np.log1p(np.exp(-np.abs(c))) + np.maximum(c, 0.0)) / p["alpha2"]
    f_T = p["mu"] * N * np.tanh(p["alpha1"] * v_T)

    return (rfft(N, axis=1)[:, harmonics] * scale,
            rfft(f_T, axis=1)[:, harmonics] * scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="first-order result CSV, rewritten in place")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists() and (HERE / path).exists():
        path = HERE / path
    header = io.read_header(path)
    df = pd.read_csv(path, comment="#")
    if "re_h0_f_N" in df.columns:
        print(f"{path.name} already carries a force block -- nothing to do")
        return

    harmonics = sorted({int(c.split("_")[1][1:]) for c in df.columns
                        if c.startswith("re_h")})
    n_t, p = header_values(header)
    f_N, f_T = interface_force(df, harmonics, n_t, p)

    # harmonic-major, force channels after the kinematics of the same harmonic
    columns = list(df.columns[:6])
    for i, h in enumerate(harmonics):
        kinematics = [c for c in df.columns if c.startswith(("re_h", "im_h"))
                      and c.split("_")[1] == f"h{h}"]
        columns += kinematics
        for label, a in (("f_N", f_N), ("f_T", f_T)):
            df[f"re_h{h}_{label}"] = a[:, i].real
            df[f"im_h{h}_{label}"] = a[:, i].imag
            columns += [f"re_h{h}_{label}", f"im_h{h}_{label}"]

    header += [
        f"f_N, f_T: RECONSTRUCTED by {Path(__file__).name} from the stored x_rel_N,"
        f" x_rel_T harmonics with the laws above, sampled at N_t = {n_t} and",
        "  projected onto the retained harmonics; ordered [N ; f_T] as in the"
        " pyFBS AFT export",
    ]
    shutil.copy2(path, path.with_suffix(".csv.orig"))
    with open(path, "w", newline="") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        df[columns].to_csv(fh, index=False, float_format="%.10e")
    print(f"force block added to {path.name} ({len(df)} points,"
          f" {len(harmonics)} harmonics); original kept as {path.name}.orig")


if __name__ == "__main__":
    main()
