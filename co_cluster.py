"""Low-rank GW co-clustering utilities for point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import ot
import pyvista as pv


def _validate_pointcloud(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point.")
    if X.shape[1] < 3:
        raise ValueError(f"{name} must have at least 3 coordinates per point.")
    return X


def normalize_points_for_plot(
    X: np.ndarray,
    reference_points: np.ndarray | None = None,
    flip_y: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Normalize a point cloud for stable plotting."""
    X = _validate_pointcloud(X, "X")
    X_plot = np.asarray(X[:, :3], dtype=float).copy()

    if flip_y:
        X_plot[:, 1] *= -1.0

    center = X_plot.mean(axis=0)
    X_plot -= center

    scale = np.max(np.linalg.norm(X_plot, axis=1))
    if scale > 0:
        X_plot /= scale

    if reference_points is None:
        return X_plot

    ref = np.asarray(reference_points, dtype=float).copy()
    if ref.ndim != 2 or ref.shape[1] < 3:
        raise ValueError("reference_points must have shape (n_refs, d) with d >= 3.")
    ref = ref[:, :3]
    if flip_y:
        ref[:, 1] *= -1.0
    ref -= center
    if scale > 0:
        ref /= scale
    return X_plot, ref


def normalize_pointcloud(X: np.ndarray, flip_y: bool = True) -> np.ndarray:
    """Normalize a point cloud to fit nicely in the renderer."""
    return normalize_points_for_plot(X, flip_y=flip_y)  # type: ignore[return-value]


def compute_lowrank_gw_coclusters(
    X1: np.ndarray,
    X2: np.ndarray,
    rank: int,
    reg: float = 1e-2,
    numItermax: int = 1000,
    stopThr: float = 1e-7,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    random_state: int | None = 0,
    log: bool = True,
) -> dict[str, Any]:
    """Compute low-rank GW factors and hard co-cluster labels."""
    X1 = _validate_pointcloud(X1, "X1")
    X2 = _validate_pointcloud(X2, "X2")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}.")

    n1, n2 = X1.shape[0], X2.shape[0]
    a = ot.unif(n1) if a is None else np.asarray(a, dtype=float)
    b = ot.unif(n2) if b is None else np.asarray(b, dtype=float)

    if a.shape != (n1,):
        raise ValueError(f"a must have shape ({n1},), got {a.shape}.")
    if b.shape != (n2,):
        raise ValueError(f"b must have shape ({n2},), got {b.shape}.")

    if random_state is not None:
        np.random.seed(random_state)

    Q, R, g, solver_log = ot.lowrank_gromov_wasserstein_samples(
        X1,
        X2,
        a=a,
        b=b,
        reg=reg,
        rank=rank,
        numItermax=numItermax,
        stopThr=stopThr,
        log=log,
    )

    g = np.asarray(g, dtype=float)
    if np.any(g <= 0):
        raise ValueError("low-rank GW returned non-positive entries in g.")

    P_lowrank = Q @ np.diag(1.0 / g) @ R.T
    label1 = np.argmax(Q, axis=1)
    label2 = np.argmax(R, axis=1)

    return {
        "Q": Q,
        "R": R,
        "g": g,
        "log": solver_log,
        "P_lowrank": P_lowrank,
        "label1": label1,
        "label2": label2,
        "rank": rank,
        "a": a,
        "b": b,
    }


def render_cluster_pointcloud(
    X: np.ndarray,
    labels: np.ndarray,
    title: str | None = None,
    point_size: float = 18,
    window_size: tuple[int, int] = (700, 500),
    camera_position: str | list[tuple[float, float, float]] | list[float] = "iso",
    save_path: str | Path | None = None,
) -> pv.Plotter:
    """Render one clustered 3D point cloud using PyVista."""
    X_plot = normalize_pointcloud(X)
    labels = np.asarray(labels)

    colors = np.array(
        [
            [68, 119, 170],
            [238, 102, 119],
            [34, 136, 51],
            [204, 187, 68],
            [170, 51, 119],
            [102, 204, 238],
            [187, 187, 187],
            [238, 136, 102],
        ],
        dtype=np.uint8,
    )

    rgb = colors[labels % len(colors)]
    cloud = pv.PolyData(X_plot)
    cloud["cluster_color"] = rgb

    plotter = pv.Plotter(window_size=window_size, off_screen=(save_path is not None))
    plotter.set_background("white")
    plotter.add_mesh(
        cloud,
        scalars="cluster_color",
        rgb=True,
        render_points_as_spheres=True,
        point_size=point_size,
        smooth_shading=True,
    )
    plotter.enable_eye_dome_lighting()
    plotter.enable_anti_aliasing("ssaa")
    plotter.hide_axes()

    if camera_position == "iso":
        plotter.camera_position = [
            (1.7, 1.2, 0.9),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
    else:
        plotter.camera_position = camera_position

    plotter.camera.parallel_projection = True
    plotter.camera.zoom(1.25)

    if title is not None:
        plotter.add_text(title, position="upper_edge", font_size=14, color="black")

    if save_path is not None:
        plotter.screenshot(str(Path(save_path)))
        plotter.close()
    else:
        plotter.show(jupyter_backend="static")
    return plotter


def plot_coclusters_3d(
    X1: np.ndarray,
    X2: np.ndarray,
    label1: np.ndarray,
    label2: np.ndarray,
    rank: int,
    save_path: str | Path | None = None,
    point_size: float = 18,
) -> tuple[pv.Plotter, pv.Plotter]:
    """Render the two co-clustered point clouds using the paper-style PyVista view."""
    del rank
    save_base = Path(save_path) if save_path is not None else None
    save_path_1 = None if save_base is None else save_base.with_name(f"{save_base.stem}_view1{save_base.suffix}")
    save_path_2 = None if save_base is None else save_base.with_name(f"{save_base.stem}_view2{save_base.suffix}")
    plotter1 = render_cluster_pointcloud(X1, label1, title=None, point_size=point_size, save_path=save_path_1)
    plotter2 = render_cluster_pointcloud(X2, label2, title=None, point_size=point_size, save_path=save_path_2)
    return plotter1, plotter2
