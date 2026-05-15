"""Anchored 1D Wasserstein matching utilities."""

from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from scipy import sparse
from sklearn.decomposition import PCA


def emd_1d_sparse(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-15,
) -> sparse.csr_matrix:
    """Compute exact sparse 1D OT by sorting and greedy cumulative matching."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    a = np.asarray(a, dtype=float).ravel().copy()
    b = np.asarray(b, dtype=float).ravel().copy()

    if x.size != a.size:
        raise ValueError(f"x and a must have the same length, got {x.size} and {a.size}.")
    if y.size != b.size:
        raise ValueError(f"y and b must have the same length, got {y.size} and {b.size}.")
    if x.size == 0 or y.size == 0:
        raise ValueError("x and y must both be non-empty.")

    a = np.maximum(a, 0.0)
    b = np.maximum(b, 0.0)
    mass_a = float(a.sum())
    mass_b = float(b.sum())
    if mass_a <= eps or mass_b <= eps:
        raise ValueError("Input weights must have positive total mass.")
    a /= mass_a
    b /= mass_b

    order_x = np.argsort(x, kind="mergesort")
    order_y = np.argsort(y, kind="mergesort")
    a_sorted = a[order_x]
    b_sorted = b[order_y]

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    i = 0
    j = 0

    while i < x.size and j < y.size:
        while i < x.size and a_sorted[i] <= eps:
            i += 1
        while j < y.size and b_sorted[j] <= eps:
            j += 1
        if i >= x.size or j >= y.size:
            break

        mass = min(a_sorted[i], b_sorted[j])
        if mass > eps:
            rows.append(int(order_x[i]))
            cols.append(int(order_y[j]))
            data.append(float(mass))
        a_sorted[i] -= mass
        b_sorted[j] -= mass

    return sparse.coo_matrix((data, (rows, cols)), shape=(x.size, y.size)).tocsr()


def anchor_sliced_1d_coupling_for_block(
    X_block: np.ndarray,
    Y_block: np.ndarray,
    a_block: np.ndarray,
    b_block: np.ndarray,
    anchors_X: np.ndarray,
    anchors_Y: np.ndarray,
    anchor_ids: list[int] | np.ndarray | None = None,
    eps: float = 1e-15,
) -> sparse.csr_matrix:
    """Average anchor-induced 1D OT couplings inside one matched co-cluster block."""
    X_block = np.asarray(X_block, dtype=float)
    Y_block = np.asarray(Y_block, dtype=float)
    anchors_X = np.asarray(anchors_X, dtype=float)
    anchors_Y = np.asarray(anchors_Y, dtype=float)

    if X_block.ndim != 2 or Y_block.ndim != 2:
        raise ValueError("X_block and Y_block must be 2D arrays.")
    if anchors_X.shape != anchors_Y.shape:
        raise ValueError("anchors_X and anchors_Y must have the same shape.")
    if X_block.shape[1] != anchors_X.shape[1] or Y_block.shape[1] != anchors_Y.shape[1]:
        raise ValueError("Block points and anchors must live in the same dimension.")

    if anchor_ids is None:
        anchor_ids = list(range(anchors_X.shape[0]))
    anchor_ids = [int(r) for r in anchor_ids]
    if not anchor_ids:
        raise ValueError("anchor_ids must not be empty.")

    coupling_sum: sparse.csr_matrix | None = None
    for r in anchor_ids:
        dx = np.linalg.norm(X_block - anchors_X[r][None, :], axis=1)
        dy = np.linalg.norm(Y_block - anchors_Y[r][None, :], axis=1)
        P_r = emd_1d_sparse(dx, dy, a_block, b_block, eps=eps)
        coupling_sum = P_r if coupling_sum is None else coupling_sum + P_r

    assert coupling_sum is not None
    return (coupling_sum / len(anchor_ids)).tocsr()


def anchored_sliced_gw_match(
    X1: np.ndarray,
    X2: np.ndarray,
    label1: np.ndarray,
    label2: np.ndarray,
    anchors_X: np.ndarray,
    anchors_Y: np.ndarray,
    rank: int | None = None,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    anchor_mode: str = "all",
    block_mass: str = "average",
    return_dense: bool = False,
    eps: float = 1e-15,
    verbose: bool = True,
) -> dict[str, Any]:
    """Assemble the final anchored sliced coupling from co-cluster blocks."""
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    label1 = np.asarray(label1)
    label2 = np.asarray(label2)
    anchors_X = np.asarray(anchors_X, dtype=float)
    anchors_Y = np.asarray(anchors_Y, dtype=float)

    if rank is None:
        rank = int(max(label1.max(initial=0), label2.max(initial=0)) + 1)

    n1, n2 = X1.shape[0], X2.shape[0]
    a = np.ones(n1, dtype=float) / n1 if a is None else np.asarray(a, dtype=float)
    b = np.ones(n2, dtype=float) / n2 if b is None else np.asarray(b, dtype=float)

    global_rows: list[np.ndarray] = []
    global_cols: list[np.ndarray] = []
    global_data: list[np.ndarray] = []
    block_stats: list[dict[str, Any]] = []

    for k in range(rank):
        I = np.where(label1 == k)[0]
        J = np.where(label2 == k)[0]
        if len(I) == 0 or len(J) == 0:
            if verbose:
                print(f"Cluster {k}: skipped because one side is empty.")
            block_stats.append(
                {"cluster_id": k, "n_X": len(I), "n_Y": len(J), "skipped": True, "reason": "empty_side"}
            )
            continue

        a_k = a[I]
        b_k = b[J]
        mass_x = float(a_k.sum())
        mass_y = float(b_k.sum())
        if mass_x <= eps or mass_y <= eps:
            if verbose:
                print(f"Cluster {k}: skipped because block mass is too small.")
            block_stats.append(
                {"cluster_id": k, "n_X": len(I), "n_Y": len(J), "skipped": True, "reason": "small_mass"}
            )
            continue

        a_local = a_k / mass_x
        b_local = b_k / mass_y
        if anchor_mode == "all":
            anchor_ids = list(range(rank))
        elif anchor_mode == "local":
            anchor_ids = [k]
        else:
            raise ValueError("anchor_mode must be 'all' or 'local'.")

        Pk_local = anchor_sliced_1d_coupling_for_block(
            X_block=X1[I],
            Y_block=X2[J],
            a_block=a_local,
            b_block=b_local,
            anchors_X=anchors_X,
            anchors_Y=anchors_Y,
            anchor_ids=anchor_ids,
            eps=eps,
        ).tocoo()

        if block_mass == "average":
            mass_k = 0.5 * (mass_x + mass_y)
        elif block_mass == "source":
            mass_k = mass_x
        elif block_mass == "target":
            mass_k = mass_y
        elif block_mass == "min":
            mass_k = min(mass_x, mass_y)
        else:
            raise ValueError("block_mass must be 'average', 'source', 'target', or 'min'.")

        global_rows.append(I[Pk_local.row])
        global_cols.append(J[Pk_local.col])
        global_data.append(Pk_local.data * mass_k)
        block_stats.append(
            {
                "cluster_id": k,
                "n_X": len(I),
                "n_Y": len(J),
                "mass_X": mass_x,
                "mass_Y": mass_y,
                "block_mass": float(mass_k),
                "nnz": int(Pk_local.nnz),
                "anchor_ids": anchor_ids,
            }
        )
        if verbose:
            print(
                f"Cluster {k}: |X_k|={len(I)}, |Y_k|={len(J)}, "
                f"mass_X={mass_x:.6f}, mass_Y={mass_y:.6f}, block_mass={mass_k:.6f}, nnz={Pk_local.nnz}"
            )

    if global_rows:
        rows = np.concatenate(global_rows)
        cols = np.concatenate(global_cols)
        data = np.concatenate(global_data)
    else:
        rows = np.array([], dtype=int)
        cols = np.array([], dtype=int)
        data = np.array([], dtype=float)

    P_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(n1, n2)).tocsr()
    P_sparse.sum_duplicates()

    return {
        "P_sliced_sparse": P_sparse,
        "P_sliced": P_sparse.toarray() if return_dense else P_sparse,
        "block_stats": block_stats,
        "anchors_X": anchors_X,
        "anchors_Y": anchors_Y,
        "rank": rank,
        "anchor_mode": anchor_mode,
        "block_mass": block_mass,
    }


def anchored_sliced_gw_match_from_precomputed_anchors(
    X1: np.ndarray,
    X2: np.ndarray,
    label1: np.ndarray,
    label2: np.ndarray,
    anchors_X: np.ndarray,
    anchors_Y: np.ndarray,
    rank: int | None = None,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    anchor_mode: str = "all",
    block_mass: str = "average",
    return_dense: bool = False,
    eps: float = 1e-15,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compatibility alias for the refactored matching entry point."""
    return anchored_sliced_gw_match(
        X1=X1,
        X2=X2,
        label1=label1,
        label2=label2,
        anchors_X=anchors_X,
        anchors_Y=anchors_Y,
        rank=rank,
        a=a,
        b=b,
        anchor_mode=anchor_mode,
        block_mass=block_mass,
        return_dense=return_dense,
        eps=eps,
        verbose=verbose,
    )


def print_coupling_diagnostics(
    P: np.ndarray | sparse.spmatrix,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    name: str = "P",
) -> None:
    """Print sparse or dense coupling statistics."""
    if sparse.issparse(P):
        row_sum = np.asarray(P.sum(axis=1)).ravel()
        col_sum = np.asarray(P.sum(axis=0)).ravel()
        nnz = int(P.nnz)
        total_mass = float(P.sum())
    else:
        P = np.asarray(P, dtype=float)
        row_sum = P.sum(axis=1)
        col_sum = P.sum(axis=0)
        nnz = int(np.count_nonzero(P))
        total_mass = float(P.sum())

    print(f"{name}.shape = {P.shape}")
    print(f"{name}.nnz = {nnz}")
    print(f"{name}.total_mass = {total_mass:.8f}")
    if a is not None:
        print(f"{name} row marginal L1 error = {np.linalg.norm(row_sum - np.asarray(a), 1):.6e}")
    if b is not None:
        print(f"{name} col marginal L1 error = {np.linalg.norm(col_sum - np.asarray(b), 1):.6e}")


def hard_match_from_coupling(P: np.ndarray | sparse.spmatrix, direction: str = "X1_to_X2") -> np.ndarray:
    """Convert a coupling into an argmax hard matching."""
    if direction == "X1_to_X2":
        P_csr = P.tocsr() if sparse.issparse(P) else sparse.csr_matrix(P)
        match = np.full(P_csr.shape[0], -1, dtype=int)
        for i in range(P_csr.shape[0]):
            start, end = P_csr.indptr[i], P_csr.indptr[i + 1]
            if end > start:
                cols = P_csr.indices[start:end]
                vals = P_csr.data[start:end]
                match[i] = int(cols[np.argmax(vals)])
        return match
    if direction == "X2_to_X1":
        P_csc = P.tocsc() if sparse.issparse(P) else sparse.csc_matrix(P)
        match = np.full(P_csc.shape[1], -1, dtype=int)
        for j in range(P_csc.shape[1]):
            start, end = P_csc.indptr[j], P_csc.indptr[j + 1]
            if end > start:
                rows = P_csc.indices[start:end]
                vals = P_csc.data[start:end]
                match[j] = int(rows[np.argmax(vals)])
        return match
    raise ValueError("direction must be 'X1_to_X2' or 'X2_to_X1'.")


def sample_pairs_from_coupling(
    P: np.ndarray | sparse.spmatrix,
    n_pairs: int = 100,
    seed: int = 0,
    pair_mode: str = "argmax_X1_to_X2",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample index pairs from a coupling."""
    rng = np.random.default_rng(seed)
    P_sparse = P if sparse.issparse(P) else sparse.csr_matrix(P)

    if pair_mode == "argmax_X1_to_X2":
        P_csr = P_sparse.tocsr()
        row_nnz = np.diff(P_csr.indptr)
        valid_rows = np.where(row_nnz > 0)[0]
        n_sample = min(n_pairs, len(valid_rows))
        sampled_rows = rng.choice(valid_rows, size=n_sample, replace=False)
        sampled_cols = []
        for i in sampled_rows:
            start, end = P_csr.indptr[i], P_csr.indptr[i + 1]
            cols = P_csr.indices[start:end]
            vals = P_csr.data[start:end]
            sampled_cols.append(int(cols[np.argmax(vals)]))
        return sampled_rows, np.asarray(sampled_cols, dtype=int)

    if pair_mode == "mass_sample":
        P_coo = P_sparse.tocoo()
        if P_coo.nnz == 0:
            raise ValueError("The coupling has no nonzero entries.")
        probs = P_coo.data / P_coo.data.sum()
        replace = P_coo.nnz < n_pairs
        selected = rng.choice(P_coo.nnz, size=n_pairs, replace=replace, p=probs)
        return P_coo.row[selected], P_coo.col[selected]

    raise ValueError("pair_mode must be 'argmax_X1_to_X2' or 'mass_sample'.")


def sample_candidate_pairs_from_coupling(
    P: np.ndarray | sparse.spmatrix,
    n_candidates: int = 800,
    seed: int = 42,
    mode: str = "argmax_X1_to_X2",
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate matched pairs from a coupling."""
    return sample_pairs_from_coupling(P, n_pairs=n_candidates, seed=seed, pair_mode=mode)


def canonicalize_two_pointclouds(
    X1: np.ndarray,
    X2: np.ndarray,
    flip_y: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Center both clouds, apply a shared scale, then rotate to a common PCA frame."""
    X1 = np.asarray(X1[:, :3], dtype=float).copy()
    X2 = np.asarray(X2[:, :3], dtype=float).copy()

    if flip_y:
        X1[:, 1] *= -1.0
        X2[:, 1] *= -1.0

    X1 -= X1.mean(axis=0, keepdims=True)
    X2 -= X2.mean(axis=0, keepdims=True)

    scale = max(np.max(np.linalg.norm(X1, axis=1)), np.max(np.linalg.norm(X2, axis=1)), 1e-12)
    X1 /= scale
    X2 /= scale

    pca = PCA(n_components=3)
    pca.fit(np.vstack([X1, X2]))
    X1r = X1 @ pca.components_.T
    X2r = X2 @ pca.components_.T

    if np.std(X1r[:, 0]) < np.std(X1r[:, 1]):
        X1r = X1r[:, [1, 0, 2]]
        X2r = X2r[:, [1, 0, 2]]
    return X1r, X2r


def oblique_project(X: np.ndarray, depth_x: float = 0.35, depth_y: float = 0.18) -> np.ndarray:
    """Project 3D points to a stable 2D oblique view."""
    X = np.asarray(X, dtype=float)
    u = X[:, 0] + depth_x * X[:, 2]
    v = X[:, 1] + depth_y * X[:, 2]
    return np.column_stack([u, v])


def random_downsample_idx(n: int, max_points: int = 2500, seed: int = 42) -> np.ndarray:
    """Randomly subsample indices for display only."""
    rng = np.random.default_rng(seed)
    if n <= max_points:
        return np.arange(n)
    return rng.choice(n, size=max_points, replace=False)


def stratified_select_pairs(
    rows: np.ndarray,
    cols: np.ndarray,
    XY_top: np.ndarray,
    n_lines: int = 100,
    n_bins: int = 12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Spread selected pairs across the top object's horizontal extent."""
    rng = np.random.default_rng(seed)
    x = XY_top[rows, 0]
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    selected: list[int] = []
    per_bin = max(1, n_lines // n_bins)

    for b in range(n_bins):
        mask = (x >= bins[b]) & (x < bins[b + 1] if b < n_bins - 1 else x <= bins[b + 1])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        take = min(per_bin, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        selected.extend(chosen.tolist())

    selected = list(dict.fromkeys(selected))
    if len(selected) < n_lines:
        remaining = np.setdiff1d(np.arange(len(rows)), np.array(selected, dtype=int), assume_unique=False)
        if len(remaining) > 0:
            add_take = min(n_lines - len(selected), len(remaining))
            selected.extend(rng.choice(remaining, size=add_take, replace=False).tolist())

    selected_idx = np.array(selected[:n_lines], dtype=int)
    return rows[selected_idx], cols[selected_idx]


def plot_alignment_for_paper(
    X1: np.ndarray,
    X2: np.ndarray,
    P: np.ndarray | sparse.spmatrix,
    n_lines: int = 100,
    candidate_pool: int = 800,
    pair_mode: str = "argmax_X1_to_X2",
    max_points: int = 2500,
    vertical_sep: float = 2.0,
    point_size: float = 1.0,
    line_width: float = 0.9,
    line_alpha: float = 0.55,
    point_alpha: float = 0.45,
    top_color: str = "#d88c8c",
    bottom_color: str = "#6b6fc9",
    line_color: str = "#666666",
    figsize: tuple[float, float] = (7.2, 6.0),
    dpi: int = 300,
    save_path: str | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    """Produce the clean paper-style alignment figure from the original notebook."""
    X1r, X2r = canonicalize_two_pointclouds(X1, X2, flip_y=True)
    XY1 = oblique_project(X1r, depth_x=0.35, depth_y=0.18)
    XY2 = oblique_project(X2r, depth_x=0.35, depth_y=0.18)
    XY1[:, 1] += vertical_sep / 2.0
    XY2[:, 1] -= vertical_sep / 2.0

    idx1 = random_downsample_idx(len(XY1), max_points=max_points, seed=seed)
    idx2 = random_downsample_idx(len(XY2), max_points=max_points, seed=seed + 1)

    rows_cand, cols_cand = sample_candidate_pairs_from_coupling(
        P, n_candidates=candidate_pool, seed=seed, mode=pair_mode
    )
    rows, cols = stratified_select_pairs(rows_cand, cols_cand, XY1, n_lines=n_lines, n_bins=12, seed=seed)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i, j in zip(rows, cols):
        ax.plot(
            [XY1[i, 0], XY2[j, 0]],
            [XY1[i, 1], XY2[j, 1]],
            color=line_color,
            lw=line_width,
            alpha=line_alpha,
            linestyle=(0, (7, 6)),
            zorder=1,
        )

    ax.scatter(
        XY1[idx1, 0], XY1[idx1, 1], s=point_size, c=top_color, alpha=point_alpha, linewidths=0, rasterized=True, zorder=3
    )
    ax.scatter(
        XY2[idx2, 0], XY2[idx2, 1], s=point_size, c=bottom_color, alpha=point_alpha, linewidths=0, rasterized=True, zorder=3
    )
    ax.set_aspect("equal")
    ax.axis("off")

    all_xy = np.vstack([XY1[idx1], XY2[idx2]])
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    pad_x = 0.08 * (xmax - xmin + 1e-12)
    pad_y = 0.08 * (ymax - ymin + 1e-12)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    plt.tight_layout(pad=0.02)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved figure to: {save_path}")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    buffer.seek(0)
    display(Image(data=buffer.getvalue()))
    buffer.close()
    plt.show()
    return rows, cols, fig, ax


def plot_final_alignment(
    X1: np.ndarray,
    X2: np.ndarray,
    P: np.ndarray | sparse.spmatrix,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    """Convenience alias for the paper-style final alignment visualization."""
    return plot_alignment_for_paper(X1, X2, P, **kwargs)


def plot_final_coupling(
    P: np.ndarray | sparse.spmatrix,
    title: str = "Final coupling",
    figsize: tuple[float, float] = (6.2, 5.6),
    dpi: int = 220,
    marker_size: float = 1.0,
    alpha: float = 0.45,
    color: str = "#4c566a",
    invert_y: bool = True,
    save_path: str | None = None,
) -> tuple[Any, Any]:
    """Plot a dense or sparse coupling as a point-mass diagram."""
    if sparse.issparse(P):
        P_coo = P.tocoo()
        rows = P_coo.row
        cols = P_coo.col
        vals = P_coo.data
        shape = P_coo.shape
    else:
        P_arr = np.asarray(P, dtype=float)
        rows, cols = np.nonzero(P_arr)
        vals = P_arr[rows, cols]
        shape = P_arr.shape

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if len(vals) > 0:
        size_scale = marker_size
        if np.max(vals) > 0:
            size_scale = marker_size * (0.35 + 1.65 * vals / np.max(vals))
        ax.scatter(cols, rows, s=size_scale, c=color, alpha=alpha, linewidths=0, rasterized=True)

    ax.set_title(title)
    ax.set_xlabel("Target index")
    ax.set_ylabel("Source index")
    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(-0.5, shape[0] - 0.5)
    if invert_y:
        ax.invert_yaxis()
    ax.set_facecolor("white")
    ax.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved coupling figure to: {save_path}")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    buffer.seek(0)
    display(Image(data=buffer.getvalue()))
    buffer.close()
    plt.show()
    return fig, ax


def gw_value_from_distance_matrices(
    C1: np.ndarray,
    C2: np.ndarray,
    P: np.ndarray | sparse.spmatrix,
    normalize_mass: bool = True,
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Compute the GW objective value for a fixed coupling."""
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    if sparse.issparse(P):
        P_csr = P.tocsr().astype(np.float64)
        if normalize_mass:
            mass = float(P_csr.sum())
            if mass <= 0:
                raise ValueError("P has zero mass.")
            P_csr = P_csr.copy()
            P_csr.data /= mass

        r = np.asarray(P_csr.sum(axis=1)).ravel()
        c = np.asarray(P_csr.sum(axis=0)).ravel()
        term1 = float(r @ ((C1 * C1) @ r))
        term2 = float(c @ ((C2 * C2) @ c))

        P_T = P_csr.T.tocsr()
        cross = 0.0
        for start in range(0, C1.shape[0], chunk_size):
            end = min(start + chunk_size, C1.shape[0])
            C1_block = C1[start:end]
            M = (P_T @ C1_block.T).T
            N = M @ C2.T
            P_block = P_csr[start:end].tocoo()
            if P_block.nnz > 0:
                cross += float(np.dot(P_block.data, N[P_block.row, P_block.col]))
    else:
        P_dense = np.asarray(P, dtype=np.float64)
        if normalize_mass:
            mass = float(P_dense.sum())
            if mass <= 0:
                raise ValueError("P has zero mass.")
            P_dense = P_dense / mass
        r = P_dense.sum(axis=1)
        c = P_dense.sum(axis=0)
        term1 = float(r @ ((C1 * C1) @ r))
        term2 = float(c @ ((C2 * C2) @ c))
        cross = float(np.sum(P_dense * (C1 @ P_dense @ C2.T)))

    gw_raw = term1 + term2 - 2.0 * cross
    return {
        "gw_value": max(float(gw_raw), 0.0),
        "gw_raw": float(gw_raw),
        "term1": term1,
        "term2": term2,
        "cross": cross,
        "row_marginal": r,
        "col_marginal": c,
    }
