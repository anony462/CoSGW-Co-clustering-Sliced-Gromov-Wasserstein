"""Anchor selection utilities for co-clustered point clouds."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

from co_cluster import normalize_points_for_plot


def _validate_inputs(X: np.ndarray, labels: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one point.")
    if labels.shape != (X.shape[0],):
        raise ValueError(f"labels must have shape ({X.shape[0]},), got {labels.shape}.")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}.")
    if X.shape[1] < 3:
        raise ValueError("X must have at least 3 coordinates per point.")
    return X, labels


def compute_anchor_energy_medoid(X_component: np.ndarray, chunk_size: int = 512) -> tuple[int, float]:
    """Return the medoid index inside a component and its summed-distance energy."""
    X_component = np.asarray(X_component, dtype=float)
    n = X_component.shape[0]
    if n == 0:
        raise ValueError("X_component must contain at least one point.")
    if n == 1:
        return 0, 0.0

    energy = np.zeros(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        D = pairwise_distances(X_component[start:end], X_component, metric="euclidean")
        energy[start:end] = D.sum(axis=1)

    local_idx = int(np.argmin(energy))
    return local_idx, float(energy[local_idx])


def find_component_representatives(
    X: np.ndarray,
    labels: np.ndarray,
    rank: int,
    n_neighbors: int = 10,
    min_component_size: int = 1,
    chunk_size: int = 512,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Find the largest connected component and its medoid per cluster."""
    X, labels = _validate_inputs(X, labels, rank)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1.")
    if min_component_size < 1:
        raise ValueError("min_component_size must be >= 1.")

    component_info: list[dict[str, Any]] = []

    for k in range(rank):
        cluster_idx = np.where(labels == k)[0]
        cluster_size = len(cluster_idx)
        if cluster_size == 0:
            if verbose:
                print(f"Cluster {k}: empty.")
            continue

        X_cluster = X[cluster_idx, :3]
        if cluster_size == 1:
            component_info.append(
                {
                    "cluster_id": k,
                    "component_id": 0,
                    "component_indices": cluster_idx,
                    "component_size": 1,
                    "cluster_size": 1,
                    "num_components_in_cluster": 1,
                    "representative_coord": X_cluster[0].copy(),
                    "representative_idx": int(cluster_idx[0]),
                    "anchor_energy": 0.0,
                    "is_dominant": True,
                    "selected_component_size": 1,
                }
            )
            if verbose:
                print(f"Cluster {k}: size=1.")
            continue

        k_eff = min(n_neighbors, cluster_size - 1)
        graph = kneighbors_graph(X_cluster, n_neighbors=k_eff, mode="connectivity", include_self=False)
        graph = graph.maximum(graph.T)
        n_components, comp_labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        comp_sizes = np.bincount(comp_labels)

        valid_components = [c for c in range(n_components) if comp_sizes[c] >= min_component_size]
        if not valid_components:
            valid_components = [int(np.argmax(comp_sizes))]

        dominant_component = max(valid_components, key=lambda c: comp_sizes[c])
        local_idx = np.where(comp_labels == dominant_component)[0]
        global_idx = cluster_idx[local_idx]
        X_component = X[global_idx, :3]
        medoid_local_idx, energy = compute_anchor_energy_medoid(X_component, chunk_size=chunk_size)
        medoid_global_idx = int(global_idx[medoid_local_idx])

        info = {
            "cluster_id": k,
            "component_id": int(dominant_component),
            "component_indices": global_idx,
            "component_size": int(len(global_idx)),
            "cluster_size": int(cluster_size),
            "num_components_in_cluster": int(n_components),
            "component_sizes": comp_sizes.tolist(),
            "representative_coord": X[medoid_global_idx, :3].copy(),
            "representative_idx": medoid_global_idx,
            "anchor_energy": energy,
            "is_dominant": True,
            "selected_component_size": int(len(global_idx)),
        }
        component_info.append(info)

        if verbose:
            print(
                f"Cluster {k}: size={cluster_size}, components={n_components}, "
                f"largest_size={len(global_idx)}, medoid_idx={medoid_global_idx}"
            )

    return component_info


def select_one_anchor_per_cluster(
    X: np.ndarray,
    labels: np.ndarray,
    component_info: list[dict[str, Any]],
    rank: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Select one real sample anchor from component metadata for every cluster."""
    X, labels = _validate_inputs(X, labels, rank)
    anchors: list[np.ndarray] = []
    anchor_indices: list[int] = []
    selected_infos: list[dict[str, Any]] = []

    global_mean = X.mean(axis=0)
    for k in range(rank):
        candidates = [info for info in component_info if int(info["cluster_id"]) == k]
        if not candidates:
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                warnings.warn(f"Cluster {k} is empty; falling back to global mean.", stacklevel=2)
                anchors.append(global_mean.copy())
                anchor_indices.append(-1)
                selected_infos.append({"cluster_id": k, "fallback": "global_mean"})
            else:
                warnings.warn(f"Cluster {k} has no component info; using cluster medoid fallback.", stacklevel=2)
                X_cluster = X[idx]
                local_idx, energy = compute_anchor_energy_medoid(X_cluster, chunk_size=min(512, len(idx)))
                global_idx = int(idx[local_idx])
                anchors.append(X[global_idx].copy())
                anchor_indices.append(global_idx)
                selected_infos.append(
                    {
                        "cluster_id": k,
                        "fallback": "cluster_medoid",
                        "representative_idx": global_idx,
                        "anchor_energy": energy,
                        "component_indices": idx,
                    }
                )
            continue

        selected = max(candidates, key=lambda info: int(info.get("component_size", 0)))
        rep_idx = selected.get("representative_idx")
        if rep_idx is None:
            raise KeyError(f"Cluster {k} is missing representative_idx in component_info.")
        anchors.append(X[int(rep_idx)].copy())
        anchor_indices.append(int(rep_idx))
        selected_infos.append(selected)

    return np.vstack(anchors), np.asarray(anchor_indices, dtype=int), selected_infos


def compute_centroid_anchors(X: np.ndarray, labels: np.ndarray, rank: int) -> dict[str, Any]:
    """Compute one centroid anchor per co-cluster."""
    X, labels = _validate_inputs(X, labels, rank)
    anchors: list[np.ndarray] = []
    component_info: list[dict[str, Any]] = []
    global_mean = X.mean(axis=0)

    for k in range(rank):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            warnings.warn(f"Cluster {k} is empty; using global mean as centroid fallback.", stacklevel=2)
            anchor = global_mean.copy()
            component_info.append(
                {
                    "cluster_id": k,
                    "component_indices": np.array([], dtype=int),
                    "component_size": 0,
                    "cluster_size": 0,
                    "fallback": "global_mean",
                }
            )
        else:
            anchor = X[idx].mean(axis=0)
            component_info.append(
                {
                    "cluster_id": k,
                    "component_indices": idx,
                    "component_size": int(len(idx)),
                    "cluster_size": int(len(idx)),
                }
            )
        anchors.append(anchor)

    return {
        "anchors": np.vstack(anchors),
        "anchor_indices": None,
        "component_info": component_info,
        "method": "centroid",
    }


def compute_component_medoid_anchors(
    X: np.ndarray,
    labels: np.ndarray,
    rank: int,
    n_neighbors: int = 10,
    min_component_size: int = 1,
    chunk_size: int = 512,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compute one medoid anchor from the largest connected component of each cluster."""
    component_info = find_component_representatives(
        X=X,
        labels=labels,
        rank=rank,
        n_neighbors=n_neighbors,
        min_component_size=min_component_size,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    anchors, anchor_indices, selected_info = select_one_anchor_per_cluster(X, labels, component_info, rank)
    return {
        "anchors": anchors,
        "anchor_indices": anchor_indices,
        "component_info": selected_info,
        "method": "component_medoid",
    }


def compute_anchors(
    X: np.ndarray,
    labels: np.ndarray,
    rank: int,
    method: str = "component_medoid",
    n_neighbors: int = 10,
    min_component_size: int = 1,
    chunk_size: int = 512,
    verbose: bool = True,
) -> dict[str, Any]:
    """Unified anchor-selection entry point."""
    if method == "centroid":
        return compute_centroid_anchors(X, labels, rank)
    if method == "component_medoid":
        return compute_component_medoid_anchors(
            X,
            labels,
            rank,
            n_neighbors=n_neighbors,
            min_component_size=min_component_size,
            chunk_size=chunk_size,
            verbose=verbose,
        )
    raise ValueError("method must be 'centroid' or 'component_medoid'.")


def create_star_mesh(
    center: np.ndarray,
    outer_radius: float = 0.045,
    inner_radius: float = 0.020,
    thickness: float = 0.010,
    n_points: int = 5,
) -> pv.PolyData:
    """Create a small 3D star mesh for anchor rendering."""
    center = np.asarray(center, dtype=float)
    boundary = []
    for i in range(2 * n_points):
        angle = np.pi / 2 + i * np.pi / n_points
        radius = outer_radius if i % 2 == 0 else inner_radius
        boundary.append([radius * np.cos(angle), radius * np.sin(angle), 0.0])
    boundary = np.asarray(boundary)

    z_top = thickness / 2
    z_bot = -thickness / 2
    top = boundary.copy()
    bot = boundary.copy()
    top[:, 2] = z_top
    bot[:, 2] = z_bot

    top_center = np.array([[0.0, 0.0, z_top]])
    bot_center = np.array([[0.0, 0.0, z_bot]])
    points = np.vstack([top, bot, top_center, bot_center]) + center

    n = len(boundary)
    top_center_idx = 2 * n
    bot_center_idx = 2 * n + 1
    faces: list[int] = []

    for i in range(n):
        j = (i + 1) % n
        faces.extend([3, top_center_idx, i, j])
    for i in range(n):
        j = (i + 1) % n
        faces.extend([3, bot_center_idx, n + j, n + i])
    for i in range(n):
        j = (i + 1) % n
        faces.extend([4, i, n + i, n + j, j])

    return pv.PolyData(points, np.asarray(faces))


def plot_clusters_with_representatives_pretty(
    X: np.ndarray,
    labels: np.ndarray,
    component_info: list[dict[str, Any]],
    title: str = "Connected components with representatives",
    point_size: float = 11,
    point_opacity: float = 0.30,
    representative_scale: float = 0.045,
    show_labels: bool = False,
    label_prefix: str = "A",
    window_size: tuple[int, int] = (1100, 800),
    flip_y: bool = True,
    save_path: str | Path | None = None,
) -> pv.Plotter:
    """Render clustered points with gold star anchors in the original notebook style."""
    X = np.asarray(X)
    labels = np.asarray(labels)
    representative_coords = np.array([info["representative_coord"] for info in component_info], dtype=float)
    X_plot, rep_plot = normalize_points_for_plot(X, reference_points=representative_coords, flip_y=flip_y)

    colors = np.array(
        [
            [78, 121, 167],
            [242, 142, 43],
            [225, 87, 89],
            [118, 183, 178],
            [89, 161, 79],
            [237, 201, 72],
            [176, 122, 161],
            [255, 157, 167],
            [156, 117, 95],
            [186, 176, 172],
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
        opacity=point_opacity,
        smooth_shading=True,
    )

    label_points = []
    label_texts = []
    for i, info in enumerate(component_info):
        p = rep_plot[i]
        cluster_id = info.get("cluster_id", i)
        component_id = info.get("component_id", 0)
        is_dominant = info.get("is_dominant", True)

        black_star = create_star_mesh(
            center=p,
            outer_radius=representative_scale * 1.25,
            inner_radius=representative_scale * 0.52,
            thickness=representative_scale * 0.28,
        )
        plotter.add_mesh(
            black_star,
            color="black",
            smooth_shading=True,
            ambient=0.25,
            diffuse=0.8,
            specular=0.2,
            specular_power=15,
        )

        gold_star = create_star_mesh(
            center=p,
            outer_radius=representative_scale,
            inner_radius=representative_scale * 0.42,
            thickness=representative_scale * 0.34,
        )
        plotter.add_mesh(
            gold_star,
            color="#FFD700",
            smooth_shading=True,
            ambient=0.35,
            diffuse=0.8,
            specular=0.45,
            specular_power=25,
        )

        label_points.append(p)
        label_texts.append(f"{label_prefix}{cluster_id}" if is_dominant else f"{label_prefix}{cluster_id}-{component_id}")

    if show_labels and len(label_points) > 0:
        plotter.add_point_labels(
            np.array(label_points),
            label_texts,
            font_size=16,
            text_color="black",
            point_size=0,
            shape_opacity=0.0,
            always_visible=True,
            margin=3,
        )

    plotter.add_text(title, position="upper_edge", font_size=20, color="black")
    plotter.enable_eye_dome_lighting()
    plotter.enable_anti_aliasing("ssaa")
    plotter.hide_axes()
    plotter.camera_position = [
        (1.9, 1.25, 0.95),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.parallel_projection = True
    plotter.camera.zoom(1.18)

    if save_path is not None:
        plotter.screenshot(str(Path(save_path)), transparent_background=False)
        plotter.close()
    else:
        plotter.show(jupyter_backend="static")
    return plotter


def plot_anchors_3d(
    X: np.ndarray,
    labels: np.ndarray,
    anchors: np.ndarray,
    rank: int,
    title: str | None = None,
    save_path: str | Path | None = None,
    point_size: float = 10,
    point_opacity: float = 0.4,
    representative_scale: float = 0.05,
    show_labels: bool = False,
) -> pv.Plotter:
    """Compatibility wrapper that renders anchors with the pretty PyVista style."""
    X, labels = _validate_inputs(X, labels, rank)
    anchors = np.asarray(anchors, dtype=float)
    if anchors.shape[0] != rank:
        raise ValueError(f"anchors must have {rank} rows, got {anchors.shape[0]}.")

    component_info = []
    for k in range(rank):
        component_info.append(
            {
                "cluster_id": k,
                "component_id": 0,
                "component_indices": np.where(labels == k)[0],
                "component_size": int(np.sum(labels == k)),
                "representative_coord": anchors[k, :3].copy(),
                "representative_idx": None,
                "is_dominant": True,
            }
        )

    return plot_clusters_with_representatives_pretty(
        X,
        labels,
        component_info,
        title=title or "Anchors",
        point_size=point_size,
        point_opacity=point_opacity,
        representative_scale=representative_scale,
        show_labels=show_labels,
        save_path=save_path,
    )
