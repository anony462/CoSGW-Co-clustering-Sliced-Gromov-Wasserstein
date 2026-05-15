# CoSGW: Co-clustering Sliced Gromov-Wasserstein Demo

This repository contains a compact research demo for matching 3D point clouds with a two-stage pipeline:

1. Compute low-rank Gromov-Wasserstein (GW) co-clusters between two point clouds.
2. Build an anchored sliced matching inside each matched co-cluster block.

The code is organized to support both experimentation in a notebook and direct reuse from Python modules.


## Overview

The main workflow in this repo is:

1. Sample two 3D point clouds from shape meshes.
2. Run low-rank GW to obtain soft factors and hard co-cluster labels.
3. Select one anchor per cluster.
4. Compute an anchored 1D sliced coupling within each matched co-cluster block.
5. Convert the coupling into hard matches and visualize the final alignment.

This demo is currently built around CAPOD-style 3D object data and the notebook in `demo_pipeline.ipynb`.

## Repository structure

- `co_cluster.py`: low-rank GW co-clustering utilities and 3D visualization helpers.
- `anchor_selection.py`: anchor construction methods, including centroid and component-medoid anchors.
- `anchored_1d_wasserstein.py`: sparse 1D OT, anchored sliced matching, diagnostics, hard matching, and plotting.
- `demo_pipeline.ipynb`: end-to-end demo notebook.
- `data/CAPOD/`: CAPOD metadata included in this repo.

## Installation

Python 3.10+ is recommended.

Install the main dependencies with:

```bash
pip install numpy scipy scikit-learn matplotlib pot pyvista trimesh ipython notebook
```

Depending on your platform, `pyvista` may also pull in `vtk`, which can take a bit longer to install.

## Data

The notebook expects CAPOD meshes under:

```text
data/CAPOD/class{class_id}/m{sample_id}.obj
```

The repository currently includes CAPOD metadata files, but you should make sure the actual `.obj` meshes are present before running the notebook. The helper function in the notebook will raise a `FileNotFoundError` if a mesh is missing.

The included dataset note describes CAPOD as:

- 180 canonically posed 3D objects
- 15 object classes
- Wavefront `.obj` meshes

Please cite the CAPOD dataset paper if you use those data in research.

## Quick start

Open the demo notebook:

```bash
jupyter notebook demo_pipeline.ipynb
```

The notebook walks through:

1. Loading two CAPOD meshes and sampling point clouds.
2. Computing low-rank GW co-clusters.
3. Comparing centroid anchors and connected-component medoid anchors.
4. Building anchored sliced couplings.
5. Producing hard correspondences and alignment plots.


## Main API

### `co_cluster.py`

- `compute_lowrank_gw_coclusters(...)`
  Computes low-rank GW factors `Q`, `R`, weights `g`, the reconstructed low-rank coupling, and hard cluster labels.
- `plot_coclusters_3d(...)`
  Renders the two co-clustered point clouds with a paper-style view.

### `anchor_selection.py`

- `compute_anchors(..., method="centroid")`
  Uses one centroid per co-cluster.
- `compute_anchors(..., method="component_medoid")`
  Uses the medoid of the largest connected component in each co-cluster.

### `anchored_1d_wasserstein.py`

- `emd_1d_sparse(...)`
  Exact sparse 1D optimal transport via sorting and greedy cumulative matching.
- `anchored_sliced_gw_match(...)`
  Builds the final anchored sliced coupling block by block.
- `hard_match_from_coupling(...)`
  Converts a coupling into an argmax hard correspondence.
- `plot_final_alignment(...)`
  Produces the final alignment figure used in the notebook.




