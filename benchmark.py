import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from graphene_builder import (
    build_supercell,
    build_first_nn_bonds,
    k_from_grid_index,
    Hk_from_bonds_sparse,
)
from backend.engine import conformal_apply_sparse

try:
    import cupy as cp
except ImportError:
    cp = None


# =============================================================================
# GPU / CPU dense reference
# =============================================================================
def dense_reference_fd_apply(H_csr, x, beta, mu, use_gpu=True):
    """
    Dense reference apply for
        y = f_beta(H - mu I) x
    using a full Hermitian eigendecomposition.

    If CuPy is available and use_gpu=True, the eigendecomposition and filtered
    apply are performed on the GPU. The returned y_ref is always a NumPy array.

    Parameters
    ----------
    H_csr : scipy.sparse matrix
        Hermitian sparse matrix.
    x : ndarray
        Input vector.
    beta : float
        Inverse temperature.
    mu : float
        Chemical potential.
    use_gpu : bool, default=True
        Whether to use CuPy for the dense reference path.

    Returns
    -------
    y_ref : ndarray
        Dense reference action on the CPU.
    Emin : float
        Smallest eigenvalue.
    Emax : float
        Largest eigenvalue.
    t_upload : float
        Host-to-device transfer time. Zero on CPU path.
    t_eigh : float
        Eigendecomposition time.
    t_apply : float
        Time to apply the Fermi-Dirac filter in the eigenbasis.
    t_download : float
        Device-to-host transfer time. Zero on CPU path.
    backend : str
        "gpu" or "cpu".
    """
    Hd = H_csr.toarray().astype(np.complex128, copy=False)
    xh = np.asarray(x, dtype=np.complex128)

    if use_gpu and cp is not None:
        t0 = time.perf_counter()
        Hg = cp.asarray(Hd)
        xg = cp.asarray(xh)
        cp.cuda.Stream.null.synchronize()
        t_upload = time.perf_counter() - t0

        t0 = time.perf_counter()
        evals_g, V_g = cp.linalg.eigh(Hg)
        cp.cuda.Stream.null.synchronize()
        t_eigh = time.perf_counter() - t0

        Emin = float(cp.asnumpy(evals_g[0]))
        Emax = float(cp.asnumpy(evals_g[-1]))

        t0 = time.perf_counter()
        s = beta * (evals_g - mu)
        s = cp.clip(s, -700.0, 700.0)
        f = 1.0 / (1.0 + cp.exp(s))
        yg = V_g @ (f * (V_g.conj().T @ xg))
        cp.cuda.Stream.null.synchronize()
        t_apply = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_ref = cp.asnumpy(yg)
        cp.cuda.Stream.null.synchronize()
        t_download = time.perf_counter() - t0

        return y_ref, Emin, Emax, t_upload, t_eigh, t_apply, t_download, "gpu"

    t_upload = 0.0
    t_download = 0.0

    t0 = time.perf_counter()
    evals, V = np.linalg.eigh(Hd)
    t_eigh = time.perf_counter() - t0

    Emin = float(evals[0])
    Emax = float(evals[-1])

    t0 = time.perf_counter()
    s = beta * (evals - mu)
    s = np.clip(s, -700.0, 700.0)
    f = 1.0 / (1.0 + np.exp(s))
    y_ref = V @ (f * (V.conj().T @ xh))
    t_apply = time.perf_counter() - t0

    return y_ref, Emin, Emax, t_upload, t_eigh, t_apply, t_download, "cpu"


# =============================================================================
# Hamiltonian builder
# =============================================================================
def build_graphene_hk(m, n, t=-2.7):
    """
    Build the sparse k-space graphene Hamiltonian for the chosen supercell.
    """
    cell = build_supercell(m, n, a=2.46)
    bonds = build_first_nn_bonds(cell, t=t)
    kvec = k_from_grid_index(cell, 0, 0, 1, 1, gamma_centered=True)
    Hk = Hk_from_bonds_sparse(
        cell,
        bonds,
        kvec,
        onsite=0.0,
        make_hermitian=True,
    ).tocsr()
    return Hk


# =============================================================================
# One solver case against dense reference
# =============================================================================
def run_single_solver_case(
    H_csr,
    x,
    y_ref,
    beta,
    mu,
    Emin,
    Emax,
    Q,
    solver,
    tol_abs_pole,
    restart,
    maxiter_cycles,
    sdr_fixed=None,
    recycle_across_poles=False,
):
    """
    Run one conformal-apply benchmark case and compare against y_ref.
    """
    ref_norm = np.linalg.norm(y_ref) + 1e-30

    t0 = time.perf_counter()
    y, st = conformal_apply_sparse(
        H_csr=H_csr,
        x=x,
        beta=beta,
        mu=mu,
        Emin=Emin,
        Emax=Emax,
        Q=Q,
        solver=solver,
        tol_abs=tol_abs_pole,
        restart=restart,
        maxiter_cycles=maxiter_cycles,
        sdr_fixed=sdr_fixed,
        recycle_across_poles=recycle_across_poles,
    )
    elapsed = time.perf_counter() - t0
    rel_err = np.linalg.norm(y - y_ref) / ref_norm

    row = dict(
        solver=solver,
        Q=Q,
        restart=restart,
        maxiter_cycles=maxiter_cycles,
        tol_abs_pole=tol_abs_pole,
        time=elapsed,
        rel_err=rel_err,
        total_mv=st["total_mv"],
        failures=st["failures"],
        max_true_res=st["max_true_res"],
        info_bad=st.get("info_bad", 0),
        recycle_across_poles=int(recycle_across_poles),
    )

    if sdr_fixed is not None:
        for key, val in sdr_fixed.items():
            row[f"sdr_{key}"] = val

    return row


# =============================================================================
# Size and temperature sweep
# =============================================================================
def run_size_temperature_sweep(
    sizes=(5, 29, 41, 58),
    T_kelvin=(100.0, 10.0, 1.0, 0.1),
    mu=0.0,
    Q=32,
    tol_abs_pole=1e-10,
    restart=40,
    maxiter_cycles=20,
    seed=1234,
    use_gpu_ref=True,
):
    """
    Sweep system size and temperature, comparing GMRES and GMRES-SDR
    against the dense eig reference.
    """
    rows = []

    kB = 8.617333262145e-5
    T_kelvin = np.array(T_kelvin, dtype=float)
    betas = 1.0 / (kB * T_kelvin)

    sdr_fixed = {
        "ssa": 0,
        "t": 2,
        "k": 12,
        "d": 1,
        "pert": 0,
        "harmonic": 1,
        "ls_solve": "lstsq",
        "svd_tol": 1e-15,
        "verbose": 0,
    }

    rng = np.random.default_rng(seed)

    for m in sizes:
        n = m
        print(f"\n=== Size sweep: m=n={m} ===")

        H = build_graphene_hk(m, n)
        Nat = H.shape[0]
        nnz = H.nnz

        x = rng.standard_normal(Nat) + 1j * rng.standard_normal(Nat)
        x /= np.linalg.norm(x)

        for T, beta in zip(T_kelvin, betas):
            print(f"--- T = {T:.3g} K, beta = {beta:.6e} ---")

            y_ref, Emin, Emax, t_upload, t_eigh, t_ref_apply, t_download, ref_backend = dense_reference_fd_apply(
                H, x, beta, mu, use_gpu=use_gpu_ref
            )

            common = dict(
                m=m,
                n=n,
                Nat=Nat,
                nnz=nnz,
                T_kelvin=float(T),
                beta=float(beta),
                mu=float(mu),
                ref_backend=ref_backend,
                ref_upload_time=t_upload,
                ref_eigh_time=t_eigh,
                ref_apply_time=t_ref_apply,
                ref_download_time=t_download,
                Emin=Emin,
                Emax=Emax,
            )

            row_gmres = run_single_solver_case(
                H_csr=H,
                x=x,
                y_ref=y_ref,
                beta=beta,
                mu=mu,
                Emin=Emin,
                Emax=Emax,
                Q=Q,
                solver="gmres",
                tol_abs_pole=tol_abs_pole,
                restart=restart,
                maxiter_cycles=maxiter_cycles,
                sdr_fixed=None,
                recycle_across_poles=False,
            )
            row_gmres.update(common)
            rows.append(row_gmres)

            row_sdr = run_single_solver_case(
                H_csr=H,
                x=x,
                y_ref=y_ref,
                beta=beta,
                mu=mu,
                Emin=Emin,
                Emax=Emax,
                Q=Q,
                solver="sdr",
                tol_abs_pole=tol_abs_pole,
                restart=restart,
                maxiter_cycles=maxiter_cycles,
                sdr_fixed=sdr_fixed,
                recycle_across_poles=True,
            )
            row_sdr.update(common)
            rows.append(row_sdr)

            print(
                f"GMRES     | Nat={Nat:6d} | T={T:8.3g} | err={row_gmres['rel_err']:.3e} | "
                f"time={row_gmres['time']:.3f}s | mv={row_gmres['total_mv']:8d}"
            )
            print(
                f"GMRES-SDR | Nat={Nat:6d} | T={T:8.3g} | err={row_sdr['rel_err']:.3e} | "
                f"time={row_sdr['time']:.3f}s | mv={row_sdr['total_mv']:8d}"
            )

    return pd.DataFrame(rows)


# =============================================================================
# SDR ablation sweep at multiple temperatures
# =============================================================================
def run_sdr_ablation_sweep(
    m=29,
    T_kelvin=(100.0, 10.0, 1.0, 0.1),
    mu=0.0,
    Q=32,
    tol_abs_pole=1e-10,
    restart=40,
    maxiter_cycles=20,
    seed=1234,
    use_gpu_ref=True,
):
    """
    Sweep GMRES-SDR internal options at fixed system size, for multiple
    temperatures, always comparing against the same GMRES baseline.
    """
    rows = []

    kB = 8.617333262145e-5
    T_kelvin = np.array(T_kelvin, dtype=float)
    betas = 1.0 / (kB * T_kelvin)

    H = build_graphene_hk(m, m)
    Nat = H.shape[0]
    nnz = H.nnz

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(Nat) + 1j * rng.standard_normal(Nat)
    x /= np.linalg.norm(x)

    ssa_vals = [0, 1, 2]
    harmonic_vals = [0, 1]
    ls_vals = ["lstsq", "qr", "pinv"]
    recycle_vals = [False, True]
    k_vals = [6, 12, 20]
    t_vals = [2, 4]
    d_vals = [1, 2]

    for T, beta in zip(T_kelvin, betas):
        print(f"\n=== SDR ablation: m=n={m}, T={T:.3g} K ===")

        y_ref, Emin, Emax, t_upload, t_eigh, t_ref_apply, t_download, ref_backend = dense_reference_fd_apply(
            H, x, beta, mu, use_gpu=use_gpu_ref
        )

        common = dict(
            m=m,
            n=m,
            Nat=Nat,
            nnz=nnz,
            T_kelvin=float(T),
            beta=float(beta),
            mu=float(mu),
            ref_backend=ref_backend,
            ref_upload_time=t_upload,
            ref_eigh_time=t_eigh,
            ref_apply_time=t_ref_apply,
            ref_download_time=t_download,
            Emin=Emin,
            Emax=Emax,
        )

        row_gmres = run_single_solver_case(
            H_csr=H,
            x=x,
            y_ref=y_ref,
            beta=beta,
            mu=mu,
            Emin=Emin,
            Emax=Emax,
            Q=Q,
            solver="gmres",
            tol_abs_pole=tol_abs_pole,
            restart=restart,
            maxiter_cycles=maxiter_cycles,
            sdr_fixed=None,
            recycle_across_poles=False,
        )
        row_gmres.update(common)
        row_gmres["variant"] = "baseline_gmres"
        rows.append(row_gmres)

        print(
            f"baseline gmres | err={row_gmres['rel_err']:.3e} | "
            f"time={row_gmres['time']:.3f}s | mv={row_gmres['total_mv']:8d}"
        )

        for ssa, harmonic, ls_solve, recycle, k, t, d in itertools.product(
            ssa_vals, harmonic_vals, ls_vals, recycle_vals, k_vals, t_vals, d_vals
        ):
            sdr_fixed = {
                "ssa": ssa,
                "t": t,
                "k": k,
                "d": d,
                "pert": 0,
                "harmonic": harmonic,
                "ls_solve": ls_solve,
                "svd_tol": 1e-15,
                "verbose": 0,
            }

            row = run_single_solver_case(
                H_csr=H,
                x=x,
                y_ref=y_ref,
                beta=beta,
                mu=mu,
                Emin=Emin,
                Emax=Emax,
                Q=Q,
                solver="sdr",
                tol_abs_pole=tol_abs_pole,
                restart=restart,
                maxiter_cycles=maxiter_cycles,
                sdr_fixed=sdr_fixed,
                recycle_across_poles=recycle,
            )
            row.update(common)
            row["variant"] = "sdr_ablation"
            rows.append(row)

            print(
                f"sdr | T={T:8.3g} | ssa={ssa} harm={harmonic} ls={ls_solve:5s} "
                f"rec={int(recycle)} k={k:2d} t={t:2d} d={d:2d} | "
                f"err={row['rel_err']:.3e} | time={row['time']:.3f}s | mv={row['total_mv']:8d}"
            )

    return pd.DataFrame(rows)


# =============================================================================
# Plot helpers
# =============================================================================
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_size_temperature_sweep(df, outdir="plots"):
    """
    Plot GMRES vs GMRES-SDR results from the size/temperature sweep.
    Produces one set of curves per temperature.
    """
    _ensure_dir(outdir)

    temps = sorted(df["T_kelvin"].unique())
    for T in temps:
        sub = df[df["T_kelvin"] == T].copy().sort_values(["solver", "Nat"])
        gm = sub[sub["solver"] == "gmres"].sort_values("Nat")
        sdr = sub[sub["solver"] == "sdr"].sort_values("Nat")

        tag = f"T_{T:g}K".replace(".", "p")

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(gm["Nat"], gm["time"], marker="o", label="GMRES")
        ax.plot(sdr["Nat"], sdr["time"], marker="s", label="GMRES-SDR")
        ax.set_xlabel("System size N")
        ax.set_ylabel("Wall time [s]")
        ax.set_title(f"Time vs N at T = {T:g} K")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{outdir}/size_time_{tag}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(gm["Nat"], gm["total_mv"], marker="o", label="GMRES")
        ax.plot(sdr["Nat"], sdr["total_mv"], marker="s", label="GMRES-SDR")
        ax.set_xlabel("System size N")
        ax.set_ylabel("Total matrix-vector products")
        ax.set_title(f"Total matvecs vs N at T = {T:g} K")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{outdir}/size_matvecs_{tag}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.semilogy(gm["Nat"], gm["rel_err"], marker="o", label="GMRES")
        ax.semilogy(sdr["Nat"], sdr["rel_err"], marker="s", label="GMRES-SDR")
        ax.set_xlabel("System size N")
        ax.set_ylabel("Relative error vs dense eig reference")
        ax.set_title(f"Accuracy vs N at T = {T:g} K")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{outdir}/size_error_{tag}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(gm["Nat"], gm["failures"], marker="o", label="GMRES")
        ax.plot(sdr["Nat"], sdr["failures"], marker="s", label="GMRES-SDR")
        ax.set_xlabel("System size N")
        ax.set_ylabel("Failed pole solves")
        ax.set_title(f"Failures vs N at T = {T:g} K")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{outdir}/size_failures_{tag}.png", dpi=200)
        plt.close(fig)

        merged = gm[["Nat", "time", "total_mv"]].merge(
            sdr[["Nat", "time", "total_mv"]],
            on="Nat",
            suffixes=("_gmres", "_sdr"),
        )
        merged["time_speedup"] = merged["time_gmres"] / merged["time_sdr"]
        merged["mv_speedup"] = merged["total_mv_gmres"] / merged["total_mv_sdr"]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(merged["Nat"], merged["time_speedup"], marker="o", label="time speedup")
        ax.plot(merged["Nat"], merged["mv_speedup"], marker="s", label="matvec speedup")
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xlabel("System size N")
        ax.set_ylabel("GMRES / SDR")
        ax.set_title(f"Speedup vs N at T = {T:g} K")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{outdir}/size_speedup_{tag}.png", dpi=200)
        plt.close(fig)


def plot_sdr_ablation(df, outdir="plots"):
    """
    Plot SDR ablation results for each temperature.
    """
    _ensure_dir(outdir)

    temps = sorted(df["T_kelvin"].unique())
    for T in temps:
        subT = df[df["T_kelvin"] == T].copy()
        gm = subT[subT["solver"] == "gmres"].copy()
        sdr = subT[subT["solver"] == "sdr"].copy()

        if len(gm) == 0 or len(sdr) == 0:
            continue

        gm_time = gm["time"].iloc[0]
        gm_mv = gm["total_mv"].iloc[0]
        gm_err = gm["rel_err"].iloc[0]

        sdr["time_speedup_vs_gmres"] = gm_time / sdr["time"]
        sdr["mv_speedup_vs_gmres"] = gm_mv / sdr["total_mv"]
        sdr["err_ratio_vs_gmres"] = sdr["rel_err"] / max(gm_err, 1e-30)

        tag = f"T_{T:g}K".replace(".", "p")

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        for ssa in sorted(sdr["sdr_ssa"].dropna().unique()):
            sub = sdr[sdr["sdr_ssa"] == ssa]
            ax.scatter(sub["time"], sub["rel_err"], label=f"ssa={int(ssa)}", alpha=0.8)

        ax.scatter([gm_time], [gm_err], marker="*", s=180, label="GMRES baseline")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wall time [s]")
        ax.set_ylabel("Relative error vs dense eig reference")
        ax.set_title(f"SDR ablation: time vs accuracy at T = {T:g} K")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{outdir}/ablation_time_vs_error_{tag}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.scatter(sdr["time_speedup_vs_gmres"], sdr["err_ratio_vs_gmres"], alpha=0.8)
        ax.axvline(1.0, linestyle="--", linewidth=1)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time speedup vs GMRES")
        ax.set_ylabel("Error ratio vs GMRES")
        ax.set_title(f"SDR variants vs GMRES at T = {T:g} K")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{outdir}/ablation_speedup_vs_errorratio_{tag}.png", dpi=200)
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    T_kelvin = np.array([100.0, 10.0, 1.0, 0.1], dtype=float)
    kB = 8.617333262145e-5
    betas = 1.0 / (kB * T_kelvin)
    mu = 0.0

    print("Temperature sweep:")
    for T, beta in zip(T_kelvin, betas):
        print(f"  T = {T:8.3g} K   beta = {beta:.6e}")

    df_size = run_size_temperature_sweep(
        sizes=(5, 29, 41, 58),
        T_kelvin=T_kelvin,
        mu=mu,
        Q=32,
        tol_abs_pole=1e-10,
        restart=40,
        maxiter_cycles=20,
        seed=1234,
        use_gpu_ref=True,
    )
    df_size.to_csv("benchmark_size_temperature_sweep.csv", index=False)

    df_ablation = run_sdr_ablation_sweep(
        m=29,
        T_kelvin=T_kelvin,
        mu=mu,
        Q=32,
        tol_abs_pole=1e-10,
        restart=40,
        maxiter_cycles=20,
        seed=1234,
        use_gpu_ref=True,
    )
    df_ablation.to_csv("benchmark_sdr_ablation_temperature.csv", index=False)

    plot_size_temperature_sweep(df_size, outdir="plots")
    plot_sdr_ablation(df_ablation, outdir="plots")

    print("\nSaved:")
    print("  benchmark_size_temperature_sweep.csv")
    print("  benchmark_sdr_ablation_temperature.csv")
    print("  plots/")


if __name__ == "__main__":
    main()