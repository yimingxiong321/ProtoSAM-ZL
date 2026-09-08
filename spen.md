# SPEN: Self-guided Prototype ENhancement for ALPNet

## Motivation

Ablation results show ALPNet's coarse mask quality is the primary bottleneck
(85% of the 10.4% gap to oracle). Specifically FN errors: organ regions not
activated (Recall 85.45% vs oracle 96.71%). The current prototype matching uses
a fixed grid (8x8) with uniform weighting — all local prototypes contribute
equally regardless of query relevance.

## Design (inspired by SPENet, adapted to training-free setting)

ProtoSAM freezes DINOv2 at inference (no training). Both SPENet modules we
adapt are training-free:

### Module 1: Adaptive Local Prototype Generation (ALPG)

Current: Fixed 8x8 grid via AvgPool2d. Each cell becomes one prototype.
Problem: Small organs get few prototypes; large organs get too many.

New: Farthest point sampling (FPS) on foreground features to find k adaptive
cluster centers, where k = min(max(mask_area / Cs, 1), k_max). Each center
defines a local region via nearest-neighbor assignment; region features are
averaged to form local prototypes.

### Module 2: Query-guided Local Prototype Enhancement (QLPE)

Current: All local prototypes contribute equally via softmax aggregation.
Problem: Support prototypes irrelevant to the current query still influence
the prediction, causing FP activation.

New: After generating support local prototypes, also generate query local
prototypes (using global-prototype-based coarse prediction as pseudo-mask).
Compute similarity matrix S between support and query local prototypes.
Solve optimal transport T* via Sinkhorn (10 iterations, no training needed).
Weight each support prototype by W* = sum(T* * S, axis=1). Combine with
global prototype: p_fused = p_global + weighted_sum(p_local * W*).

## Pipeline

```
Current:
  DINOv2 -> supp_fts, qry_fts
  -> AvgPool2d(8x8) -> fixed grid prototypes
  -> cosine_sim(qry_fts, prototypes) -> softmax -> pred

New (SPEN):
  DINOv2 -> supp_fts, qry_fts
  -> ALPG: FPS on supp_fg_fts -> k local prototypes + 1 global prototype
  -> ALPG on qry: global_proto * qry_fts -> coarse qry mask -> k_q qry prototypes
  -> QLPE: sim_matrix(supp_local, qry_local) -> Sinkhorn OT -> weights W*
  -> fuse: p_fused = p_global + sum(p_local_i * W_i)
  -> cosine_sim(qry_fts, p_fused) -> pred
```

## Implementation Strategy

Add a new file `models/spen.py` containing:
1. `adaptive_local_prototypes(fts, mask, k_max=16)` — FPS + clustering
2. `sinkhorn_ot(cost_matrix, n_iter=10)` — Sinkhorn algorithm
3. `query_guided_weighting(supp_local, qry_local)` — OT-based weighting
4. `SPENProtoMatcher` — wraps the full pipeline, replaces cls_unit

Patch `grid_proto_fewshot.py` to use SPEN when `cls_name == 'spen'`.
No changes to ProtoSAM.py or validation_protosam.py needed.

## Exploration Points (one per experiment)

### Experiment 1: ALPG only (no QLPE)
- EP: Does adaptive prototype count improve Recall?
- Replace fixed grid with FPS-based adaptive prototypes
- Keep uniform weighting (no OT)
- Success if Dice > 80% (beats baseline 79.22%)

### Experiment 2: ALPG + QLPE (full SPEN)
- EP: Does query-guided weighting further improve over ALPG alone?
- Add Sinkhorn OT weighting on top of Experiment 1
- Success if Dice > Experiment 1 result

### Experiment 3: QLPE only (fixed grid + OT weighting)
- EP: Is OT weighting useful even with fixed grid?
- Keep 8x8 grid, but add OT-based weighting
- Success if Dice > baseline

## Config

```bash
# In run_ablation.sh or run_protosam.sh, change:
clsname=spen          # instead of grid_proto
proto_grid_size=8     # used as k_max for ALPG
```

## Run

```bash
cd /share/home/huafuchen01/zl/R2Seg-main/ProtoSAM-main/
python3 outputs/patch_spen.py

# Experiment 1: ALPG only
./outputs/run_ablation.sh spen_alpg liver mri 1 "" 0

# Experiment 2: ALPG + QLPE
./outputs/run_ablation.sh spen_full liver mri 1 "" 0

# Experiment 3: QLPE only
./outputs/run_ablation.sh spen_qlpe liver mri 1 "" 0
```
