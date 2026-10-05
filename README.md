# ProtoSAM - One shot segmentation with foundational models

Link to our paper [here](https://arxiv.org/abs/2407.07042). \
This work is the successor of [DINOv2-based-Self-Supervised-Learning](https://github.com/levayz/DINOv2-based-Self-Supervised-Learning) (Link to [Paper](arxiv.org/abs/2403.03273)).

## Abstract
This work introduces a new framework, ProtoSAM, for one-shot image segmentation. It combines DINOv2, a vision transformer that extracts features from images, with an Adaptive Local Prototype Pooling (ALP) layer, which generates prototypes from a support image and its mask. These prototypes are used to create an initial coarse segmentation mask by comparing the query image's features with the prototypes.
Following the extraction of an initial mask, we use numerical methods to generate prompts, such as points and bounding boxes, which are then input into the Segment Anything Model (SAM), a prompt-based segmentation model trained on natural images. This allows segmenting new classes automatically and effectively without the need for additional training. 

## 相对原版 ProtoSAM 的扩展（QSPA）

在 **不改变 SAM prompt 提取与推理** 的前提下，本仓库在 coarse 阶段增加了 **Query-conditioned Support Prototype Aggregation（QSPA）** 及可复现实验协议。实现见 `models/qspa.py`、`models/grid_proto_fewshot.py`（Scheme A 融合）、`validation_protosam.py`；DINO backbone 函数式封装见 `models/dino_backbone.py`（默认 **DINOv2** 不变，可选 **`modelname=dinov3_l16` / `dinov3_h16`** → 本地 HF 目录 `pretrained_model/dinov3-vitl16-pretrain-lvd1689m/`、`pretrained_model/dinov3-vith16plus-pretrain-lvd1689m/`，由 Meta `.pth` 生成见 `scripts/build_dinov3_hf_snapshot.py --variant vitl16|vith16plus`）；Polyp / 3D 辅助见 `models/polyp_support.py`、`models/alp_support.py`；一键评测见 `official_benchmark_rerun/`。

### 方法改进（三点）

1. **QSPA + Scheme A（多 support 粗分割融合）**  
   从 support pool 中按 query 用 DINO 相似度取 Top-K，对相似度做 temperature-scaled softmax 得到权重；**每个 support 仍独立** 构造 fg/bg（及 local grid）prototype 并生成 similarity map，再按权重 **对 map 加权求和**（不对跨 support 的 local prototype 做向量相加）。Coarse 之后的 CCA、box/point 提取不变；精修阶段可选用 **SAM 1**（`protosam_sam_ver=sam_h`）或 **SAM 3**（`protosam_sam_ver=sam3`，权重见 `sam3_checkpoint`，适配代码 `models/sam3_predictor.py`）。

2. **统一的 support 检索与选择框架**  
   配置项 `support_selection`：`random`（原版随机 baseline）、`top1`（DINO 检索单张，不走 QSPA 加权融合）、`topk_weighted`（QSPA）。检索打分 `support_retrieval_mode`：`gap`（全局 embedding 余弦）或 `spatial`（空间特征图相似度再池化）。相关超参：`top_k`、`prototype_temperature`（默认 0.07），见 `config_ssl_upload.py`。

3. **可复现的 Polyp / 3D support 协议**  
   - **Matched**：`polyp_match_support_to_query=True` 时，query 只从 **同一 Polyp 子数据集** 的训练池检索（如 Kvasir 900 / Clinic 550）。  
   - **Colon / ETIS**：无官方 train pool 时，使用 **9:1** 划分（`official_benchmark_rerun/create_colon_etis_91_split.py`，`polyp_colon_etis_split_dir`）。  
   - **3D CT/MRI**：从官方 support scan 构建 slice pool（`models/alp_support.py`），同一套 QSPA 接入验证脚本。

### Polyp 2D 实验结果

**Kvasir**（100 test）+ **CVC-ClinicDB**（62 test）：matched 同子集 support 池。**CVC-ColonDB**（38 test）+ **ETIS-LaribPolypDB**（20 test）：**9:1** support/test（`create_colon_etis_91_split.py`）。`top1` = DINO **GAP** 单 support；`k3`/`k5` = **QSPA**（`topk_weighted`）。**All Dice / All IoU**：有四子集时用 **100:62:38:20** 加权；缺 Colon/ETIS 时用 **100:62**（仅 Kvasir+Clinic）。SAM1 精修为 **bbox + Conf/Cent 点**（`point_mode=both`，与 ProtoSAM 默认一致）。

#### SAM 1 Protosam Baseline


| K | 检索 | All Dice | All IoU | Kvasir Dice | Kvasir IoU | Clinic Dice | Clinic IoU | Colon Dice | Colon IoU | ETIS Dice | ETIS IoU |
|---|------|----------|---------|-------------|------------|-------------|------------|------------|-----------|-----------|----------|
| 1 | GAP | 79.66 | 72.15 | 72.96 | 65.07 | 85.98 | 79.62 | 83.09 | 74.65 | 87.10 | 79.59 |
| 3 | GAP | 84.53 | 77.25 | 81.63 | 74.10 | 85.82 | 79.35 | 88.08 | 79.97 | 88.34 | 81.32 |
| 5 | GAP | 83.95 | 76.90 | 82.31 | 75.09 | 85.12 | 78.67 | 83.96 | 76.35 | 88.46 | 81.52 |
| 3 | spatial | 84.04 | 76.59 | 81.55 | 73.56 | 87.16 | 80.59 | 82.92 | 75.16 | 88.95 | 82.04 |
| 5 | spatial | 85.09 | 77.66 | 82.41 | 74.73 | 87.32 | 80.74 | 86.90 | 78.67 | 88.13 | 80.84 |

#### 对比：DINOv2-L/14 vs DINOv3-H+（SAM1 精修不变）


| K | 检索 | DINOv2 All Dice | DINOv3 H+ All Dice | Δ | DINOv2 All IoU | DINOv3 H+ All IoU | Δ |
|---|------|-------------|-------------|---|------------|------------|---|
| 1 | GAP | 79.66 | **80.89** | +1.23 | 72.15 | **73.33** | +1.18 |
| 3 | GAP | 84.53 | **83.84** | −0.69 | 77.25 | **76.58** | −0.67 |
| 5 | GAP | 83.95 | **83.45** | −0.50 | 76.90 | **76.56** | −0.34 |
| 3 | spatial | 84.04 | **84.38** | +0.34 | 76.59 | **77.47** | +0.88 |
| 5 | spatial | 85.09 | **83.61** | −1.48 | 77.66 | **76.88** | −0.78 |

分数据集 **Dice**（每格为 **D2 / H+ / Δ**，单位百分点）：

| K | 检索 | Kvasir | Clinic | Colon | ETIS |
|---|------|--------|--------|-------|------|
| 1 | GAP | 72.96 / **82.62** / +9.66 | 85.98 / 77.74 / −8.24 | 83.09 / 79.95 / −3.14 | 87.10 / 83.75 / −3.35 |
| 3 | GAP | 81.63 / **85.64** / +4.01 | 85.82 / 80.28 / −5.54 | 88.08 / 81.70 / −6.38 | 88.34 / **89.99** / +1.65 |
| 5 | GAP | 82.31 / 84.11 / +1.80 | 85.12 / 82.12 / −3.00 | 83.96 / 81.94 / −2.02 | 88.46 / 87.15 / −1.31 |
| 3 | spatial | 81.55 / 84.75 / +3.20 | 87.16 / 84.69 / −2.47 | 82.92 / 81.78 / −1.14 | 88.95 / 86.56 / −2.39 |
| 5 | spatial | 82.41 / 84.15 / +1.74 | 87.32 / 85.04 / −2.28 | 86.90 / 82.15 / −4.75 | 88.13 / 79.23 / −8.90 |

**DINOv3-H+** 绝对值（与 D2 SAM1 表同列）：

| K | 检索 | All Dice | All IoU | Kvasir Dice | Clinic Dice | Colon Dice | ETIS Dice |
|---|------|----------|---------|-------------|-------------|------------|-----------|
| 1 | GAP | 80.89 | 73.33 | 82.62 | 77.74 | 79.95 | 83.75 |
| 3 | GAP | 83.84 | 76.58 | 85.64 | 80.28 | 81.70 | 89.99 |
| 5 | GAP | 83.45 | 76.56 | 84.11 | 82.12 | 81.94 | 87.15 |
| 3 | spatial | 84.38 | 77.47 | 84.75 | 84.69 | 81.78 | 86.56 |
| 5 | spatial | 83.61 | 76.88 | 84.15 | 85.04 | 82.15 | 79.23 |

简要结论：H+ 在 **Kvasir** 上多数高于 D2；**Clinic / Colon** 多数略低。**加权 All**：**top1 GAP**（+1.23 Dice）、**spatial k=3**（+0.34）优于 D2；**spatial k=5** 仍低约 1.5 Dice（ETIS 79.23 拉低明显）。top1 上 Kvasir↑、Clinic↓ 幅度大，与 D2 检索/粗分割差异需单独分析。

#### Coarse backbone 对比：DINOv3 ViT-L/16 vs DINOv3-H+（`dinov3_l16` vs `dinov3_h16`，SAM1 精修不变）

**DINOv3-L 绝对值**（与 D2 / H+ SAM1 表同列）：

| K | 检索 | All Dice | All IoU | Kvasir Dice | Clinic Dice | Colon Dice | ETIS Dice |
|---|------|----------|---------|-------------|-------------|------------|-----------|
| 1 | GAP | 80.92 | 73.76 | 80.43 | 81.79 | 78.98 | 84.40 |
| 3 | GAP | 83.13 | 76.07 | 82.96 | 85.45 | 78.60 | 85.38 |
| 5 | GAP | 83.16 | 76.28 | 83.50 | 83.30 | 81.56 | 84.00 |
| 3 | spatial | 83.84 | 76.80 | 84.28 | 85.24 | 79.65 | 85.23 |
| 5 | spatial | 83.83 | 76.96 | 84.64 | 85.39 | 79.93 | 82.39 |

**L vs H+**（Δ = H+ − L，单位百分点）：

| K | 检索 | L All Dice | H+ All Dice | Δ | L All IoU | H+ All IoU | Δ |
|---|------|------------|-------------|---|-----------|------------|---|
| 1 | GAP | **80.92** | 80.89 | −0.03 | **73.76** | 73.33 | −0.43 |
| 3 | GAP | 83.13 | **83.84** | +0.72 | 76.07 | **76.58** | +0.51 |
| 5 | GAP | 83.16 | **83.45** | +0.29 | 76.28 | **76.56** | +0.28 |
| 3 | spatial | 83.84 | **84.38** | +0.55 | 76.80 | **77.47** | +0.67 |
| 5 | spatial | **83.83** | 83.61 | −0.22 | **76.96** | 76.88 | −0.08 |

分数据集 **Dice**（每格 **L / H+ / Δ**）：

| K | 检索 | Kvasir | Clinic | Colon | ETIS |
|---|------|--------|--------|-------|------|
| 1 | GAP | 80.43 / **82.62** / +2.19 | **81.79** / 77.74 / −4.05 | 78.98 / **79.95** / +0.97 | **84.40** / 83.75 / −0.65 |
| 3 | GAP | 82.96 / **85.64** / +2.68 | **85.45** / 80.28 / −5.17 | 78.60 / **81.70** / +3.10 | 85.38 / **89.99** / +4.61 |
| 5 | GAP | 83.50 / **84.11** / +0.60 | **83.30** / 82.12 / −1.18 | 81.56 / **81.94** / +0.38 | 84.00 / **87.15** / +3.15 |
| 3 | spatial | 84.28 / **84.75** / +0.46 | **85.24** / 84.69 / −0.54 | 79.65 / **81.78** / +2.14 | 85.23 / **86.56** / +1.34 |
| 5 | spatial | **84.64** / 84.15 / −0.49 | **85.39** / 85.04 / −0.35 | 79.93 / **82.15** / +2.22 | **82.39** / 79.23 / −3.16 |

简要结论：**L 与 H+ 加权 All 接近**（top1 略优 L，k3/k5 GAP 与 spatial k3 略优 H+，spatial k5 All 略优 L）。**Clinic（matched）上 L 五组均高于 H+**（约 0.4–5.2 Dice）；**Kvasir / Colon / ETIS（k3 GAP）** H+ 更高。**spatial k5 ETIS** L 明显优于 H+（82.39 vs 79.23），是 L 在 All 上反超 H+ 的主因。相对 D2，**L top1 All（80.92）** 介于 D2（79.66）与 H+（80.89）之间，且 **Kvasir–Clinic 更均衡**（无 H+ 式 Clinic 大幅回落）。

#### 精修 backbone：SAM 3（`MedicalSAM3/checkpoint/sam3.pt`）


**精修提示：text `"polyp"` + coarse bbox（T+I，无 Conf/Cent 点）**

| K | 检索 | All Dice | All IoU | Kvasir Dice | Kvasir IoU | Clinic Dice | Clinic IoU | Colon Dice | Colon IoU | ETIS Dice | ETIS IoU |
|---|------|----------|---------|-------------|------------|-------------|------------|------------|-----------|-----------|----------|
| 1 | GAP | 76.18 | 68.46 | 69.17 | 61.62 | 80.69 | 73.37 | 82.35 | 73.60 | 85.56 | 77.70 |
| 3 | GAP | 82.87 | 75.08 | 79.26 | 71.44 | 84.05 | 76.71 | 88.09 | 79.73 | 87.34 | 79.43 |
| 5 | GAP | 81.85 | 74.30 | 79.88 | 72.07 | 81.60 | 74.50 | 84.17 | 76.64 | 88.04 | 80.40 |
| 3 | spatial | 82.15 | 74.25 | 78.14 | 69.70 | 85.85 | 78.64 | 83.28 | 75.63 | 88.56 | 80.82 |
| 5 | spatial | 82.76 | 75.02 | 79.47 | 71.61 | 84.41 | 77.26 | 86.25 | 77.91 | 87.47 | 79.66 |

结论：整体比ProtoSAM 低了2个点。但这不太公平，因为protosam的prompt是bbox和点（sam1不接受 text prompt）

**精修提示：T+I + Conf/Cent 前景点**

| K | 检索 | All Dice | All IoU | Kvasir Dice | Kvasir IoU | Clinic Dice | Clinic IoU | Colon Dice | Colon IoU | ETIS Dice | ETIS IoU |
|---|------|----------|---------|-------------|------------|-------------|------------|------------|-----------|-----------|----------|
| 1 | GAP | 78.65 | 71.05 | 72.00 | 64.19 | 83.44 | 76.68 | 83.72 | 75.26 | 87.45 | 79.95 |
| 3 | GAP | 84.01 | 76.48 | 80.64 | 72.66 | 85.20 | 78.70 | 88.57 | 80.50 | 88.51 | 81.21 |
| 5 | GAP | 83.03 | 75.69 | 81.46 | 73.57 | 83.06 | 76.71 | 84.25 | 76.73 | 88.45 | 81.10 |
| 3 | spatial | 83.26 | 75.66 | 79.71 | 71.34 | 86.74 | 80.20 | 83.69 | 76.18 | 89.36 | 82.22 |
| 5 | spatial | 84.04 | 76.52 | 80.63 | 72.75 | 86.53 | 79.90 | 86.82 | 78.77 | 88.05 | 80.62 |

结论：加上点提示后，prompt相比 protosam 多了 text prompt，此时整体比使用 sam1 的protosam低了 0.5 个点左右

> **SAM3 接入（2026-10）**：`models/sam3_grounding.py` — MedSAM3 式 **T+I**（`text="polyp"` + coarse **bbox**）；可选将 ProtoSAM 的 **Conf/Cent** 写入 `input_points`（`SAM3_USE_COARSE_POINTS=1`）。**SAM1 仍走 `SamPredictor`**。Matched 重跑：`launch_polyp_sam3_parallel.sh`。

简要结论（SAM1，四集合并 All）：**top1 相对 QSPA（k≥3）在 All 上约低 4–5 个 Dice 点**；Colon 上 **k5 spatial QSPA 明显优于 top1 / k3**，top1 与 k3 接近；ETIS 上 **k3 spatial 略优于 k5**。

### 总结
1.引入dinov3替代v2整体并没有带来结果的提升。

2.引入sam3取代sam1反而整体结果下降了，而且是在比sam1多了text prompt的情况下。



## How To Run
### 1. Data preprocessing
#### 1.1 CT and MRI Dataset
Please see the notebook `data/data_processing.ipynb` for instructions.
For convenience i've compiled the data processing instructions from https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation to a single notebook.  \
The CT dataset is available here: https://www.synapse.org/Synapse:syn3553734 \
The MRI dataset is availabel here: https://chaos.grand-challenge.org

run `./data/CHAOST2/dcm_img_to_nii.sh` to convert dicom images to nifti files.

#### 1.2 Polyp Dataset
Data is available here: https://www.kaggle.com/datasets/hngphmv/polypdataset?select=train.csv

Put the dataset `data/PolypDataset/`

### 2. Running
#### 2.1 (Optional) Training and Validation of the coarse segmentation networks
```
./backbone.sh [MODE] [MODALITY] [LABEL_SET]
```
MODE - validation or training \
MODALITY - ct or mri \
LABEL_SET - 0 (kidneys), 1 (liver spleen)

for example:
```
./backbone.sh training mri 1
```
Please refer to `backbone.sh` for further configurations.

#### 2.1 Running ProtoSAM
Put all SAM checkpoint like sam_vit_b.pth, sam_vit_h.pth, medsam_vit_b.pth into the `pretrained_model` directory. \
Checkpoints are available at [SAM](https://github.com/facebookresearch/segment-anything) and [MedSAM](https://github.com/bowang-lab/MedSAM). \
For **SAM 3** polyp/QSPA runs set `protosam_sam_ver=sam3` and `sam3_checkpoint=/path/to/sam3.pt` (default points to `MedicalSAM3/checkpoint/sam3.pt` on our cluster).

**DINOv3 ViT-L/16 / ViT-H+ (`modelname=dinov3_l16` | `dinov3_h16`)**: place Meta `.pth` under `pretrained_model/`, run `python scripts/build_dinov3_hf_snapshot.py --variant vitl16` or `--variant vith16plus` to create the HF dirs (defaults in `config_ssl_upload.py`). Requires `pip install -r requirements-dinov3.txt`. Smoke: `python scripts/smoke_dinov3_backbone.py`.

```
./run_protosam.sh [MODALITY] [LABEL_SET]
```
MODALITY - ct, mri or polyp \
LABEL_SET (only relevant if doing ct or mri) - 0 (kidneys), 1 (liver spleen) 
Please refer to the `run_protosam.sh` script for further configurations.


## FAME-BraTS 协议口径（D6 / D7）

FAME 实验在 `fame_protosam/`。只看本仓库根 README 时，BraTS 数字按三层读，不要把 Table 2 的 299 当成某个 task 的 episode 分母，也不要把 ET/TC/WT 合并成一个 `Brain tumor` task。

```text
dataset/source level: BraTS = 299 (237 / 62)
task level:           ET / TC / WT
episode level:        每 task 50 support + 62 fixed test（EXP00B）

D6 = reporting_mismatch
D7 = true    # task-granularity difference, not an implementation error

EXP05  = official released episode
EXP00B = paper-style disjoint fixed-test
```

完整判定见 `fame_protosam/README.md` 与 `fame_protosam/FAME_PROTOCOL_NOTES.md`。

## Acknowledgements
This work is largely based on [ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation), [DINOv2](https://github.com/facebookresearch/dinov2), [SAM](https://github.com/facebookresearch/segment-anything) and is a continuation of [DINOv2-based-Self-Supervised-Learning](https://github.com/levayz/DINOv2-based-Self-Supervised-Learning).

## Cite
If you found this repo useful, please consider giving us a citation and a star!

```bibtex
@article{ayzenberg2024protosam,
  title={ProtoSAM-One Shot Medical Image Segmentation With Foundational Models},
  author={Ayzenberg, Lev and Giryes, Raja and Greenspan, Hayit},
  journal={arXiv preprint arXiv:2407.07042},
  year={2024}
}

@misc{ayzenberg2024dinov2,
      title={DINOv2 based Self Supervised Learning For Few Shot Medical Image Segmentation}, 
      author={Lev Ayzenberg and Raja Giryes and Hayit Greenspan},
      year={2024},
      eprint={2403.03273},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
