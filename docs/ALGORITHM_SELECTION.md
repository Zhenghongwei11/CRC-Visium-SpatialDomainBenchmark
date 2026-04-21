# Algorithm selection + predeclared evaluation protocol

This file is the “protocol lock”. Any change that affects conclusions MUST be tracked via OpenSpec.

## 1) What we are benchmarking
- Task: **spatial domain identification / clustering** on spot-based Visium data.
- Output: per-spot domain labels + optional domain marker lists.

## 2) Selected methods (local-first core set)
We predeclare a core set that is runnable on an 8GB laptop for the pilot:

### M0: Expression-only clustering baseline (k-means on PCs)
- Pipeline: normalization → HVG → PCA → k-means.
- Primary knobs: HVG count; fixed K.

### M1: Expression + coordinate baseline (k-means)
- Pipeline: normalization → HVG → PCA → concatenate scaled spatial coordinates → k-means.
- Primary knobs: coordinate scaling; fixed K.

### M2: BayesSpace
- Use BayesSpace’s recommended workflow for Visium when possible.
- Primary knobs: number of clusters `q` (K); MCMC/iteration settings (kept conservative for laptop).

### M3: Spatial graph baseline (Leiden)
- Pipeline: build spatial kNN graph from coordinates → edge-weight by expression similarity → Leiden partition.
- Primary knobs: neighbor count; Leiden resolution (auto-searched to match fixed K).

### M4: SpaGCN (graph convolution; xy-only, no histology)
- Pipeline: SpaGCN with histology disabled; fixed K via k-means initialisation.
- Primary knobs: fixed K; training epochs (kept conservative); distance scale `l` (auto-searched).

Cloud-only extension bucket (explicitly non-blocking for the pilot): STAGATE, SEDR, DeepST/GraphST.

## 3) Preprocessing (shared default; deviations must be recorded)
For each sample:
- Start from raw counts + spatial coordinates provided by the dataset.
- Apply minimal QC filters (predeclare thresholds; record in results/dataset_qc.tsv later).
- Use a fixed HVG cap (default: 2,000) for methods that require HVG selection.
- Standardize coordinate units (pixels vs microns) only for neighbor graph building; never change geometry.

## 4) Parameter grid (predeclared)
We will report performance as a function of K (cluster count) to avoid cherry-picking:
- K-grid: `{4, 6, 8, 10, 12}` (pilot default)
- Seeds: 20 seeds per setting (reported as distributions)

Method-specific knobs (pilot defaults; may be expanded on cloud):
- Spatial graph baseline: `k ∈ {4,6,8}`, smoothing `α ∈ {0, 0.5, 1.0}`
- BayesSpace: `q ∈ K-grid`; other parameters held at recommended defaults unless instability forces a declared exception

## 5) Evaluation metrics (predeclared; no ground truth required)
Because “true domains” are not available, we use **orthogonal** unsupervised criteria:

### 5.1 Stability
- Seed-to-seed consistency: ARI/NMI between labelings at fixed K.
- Subsampling stability: spot subsampling (e.g., 80%) then compare label transfer/alignment.
- Output table: `results/benchmarks/stability.tsv`.

### 5.2 Spatial coherence (geometry-aware)
- Neighborhood agreement / boundary smoothness metrics computed on the spatial adjacency graph.
- Output table: `results/benchmarks/spatial_coherence.tsv`.

### 5.3 Biological coherence (marker consistency)
- Within-domain marker strength and cross-sample reproducibility of marker sets.
- Output table: `results/benchmarks/marker_coherence.tsv`.

### 5.4 Compute + failure modes
- Wall time + peak RSS + explicit failure logs (OOM/timeouts/non-convergence).
- Output tables: `results/benchmarks/runtime_memory.tsv`, `results/benchmarks/failure_log.tsv`.

## 6) Negative controls (predeclared)
- **Coordinate shuffle control:** shuffle spatial coordinates and rerun spatial metrics; spatial coherence MUST degrade for spatial methods.
- **Random clustering control:** random labels matched to K as a floor baseline.
- Output table: `results/benchmarks/negative_controls.tsv`.

## 7) Fairness rules (to satisfy PLOS ONE rigor expectations)
- No post hoc tuning of parameter grids based on downstream benchmark outcomes.
- Report full distributions (not only means) and per-sample heterogeneity.
- Report failures; do not silently drop hard samples.
- Any deviation from this protocol requires an OpenSpec change.
