# CoSGW: Co-clustering + Anchored Sliced Gromov-Wasserstein Demo

This repository contains a compact research demo for matching 3D point clouds with a two-stage pipeline:

1. Compute low-rank Gromov-Wasserstein (GW) co-clusters between two point clouds.
2. Build an anchored sliced matching inside each matched co-cluster block.

The code is organized to support both experimentation in a notebook and direct reuse from Python modules.

![Example alignment](alignment_paper_final.png)

## Overview

The main workflow in this repo is:

1. Sample two 3D point clouds from shape meshes.
2. Run low-rank GW to obtain soft factors and hard co-cluster labels.
3. Select one anchor per cluster, either by centroid or by medoid of the dominant connected component.
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

## Minimal Python example

```python
import numpy as np
import ot

from co_cluster import compute_lowrank_gw_coclusters
from anchor_selection import compute_anchors
from anchored_1d_wasserstein import anchored_sliced_gw_match, hard_match_from_coupling

# X1 and X2 should be arrays of shape (n_points, d), with d >= 3
X1 = np.random.randn(1000, 3)
X2 = np.random.randn(1000, 3)

rank = 8
a = ot.unif(len(X1))
b = ot.unif(len(X2))

co_out = compute_lowrank_gw_coclusters(X1, X2, rank=rank, a=a, b=b)

anchors_X = compute_anchors(X1, co_out["label1"], rank=rank, method="component_medoid")
anchors_Y = compute_anchors(X2, co_out["label2"], rank=rank, method="component_medoid")

match_out = anchored_sliced_gw_match(
    X1=X1,
    X2=X2,
    label1=co_out["label1"],
    label2=co_out["label2"],
    anchors_X=anchors_X["anchors"],
    anchors_Y=anchors_Y["anchors"],
    rank=rank,
    a=a,
    b=b,
)

hard_match = hard_match_from_coupling(match_out["P_sliced_sparse"], direction="X1_to_X2")
print(hard_match[:10])
```

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

## Notes and assumptions

- Point clouds are expected to have at least 3 coordinates per point.
- Several visualization utilities normalize or flip coordinates for stable rendering.
- The anchored matching step works on co-cluster blocks and can use either all anchors or only the local block anchor.
- The implementation returns sparse couplings where possible to keep memory usage manageable.

## Citation

If this repository supports a paper, project report, or internal experiment, add the relevant citation here.

For CAPOD data, please cite:

```text
Panagiotis Papadakis, "The Canonically Posed 3D Objects Dataset",
Eurographics Workshop on 3D Object Retrieval (3DOR), 2014.
```

## Future improvements

- Add an explicit `requirements.txt` or `environment.yml`.
- Add a small script version of the notebook pipeline.
- Add benchmark data download instructions.
- Add unit tests for anchor selection and coupling assembly.
