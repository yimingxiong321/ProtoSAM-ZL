# Protocol B：support 选择对照（top1 vs bottom1 vs random）

## 协议设置

- **协议**：Protocol B（matched + 9:1）
- `polyp_match_support_to_query=True`：每个 query 只从同子集的 support 池选 support，不跨数据集
- Support 池：Kvasir 约 900、ClinicDB 约 550（训练集）；ColonDB 342、ETIS 176（9:1 切分）
- Test：Kvasir 100、ClinicDB 62、ColonDB 38、ETIS 20
- 加权 All 权重：100 : 62 : 38 : 20
- 特征：DINOv2-L/14，GAP 检索，1-shot，无 fusion（`polyp_colon_etis_91` 与 matched 同口径）

三种选法：

- **random**：每个 query 独立随机抽 1 张（`seed + query_index`）
- **top1**：DINO GAP 相似度最高的 1 张
- **bottom1（last1）**：DINO GAP 相似度最低的 1 张

## 结果（Dice %）

| 子集 | random | top1 | bottom1 | top1 − bottom1 |
|------|--------|------|---------|----------------|
| Kvasir | 65.85 | **72.96** | 57.44 | +15.52 |
| CVC-ClinicDB | 58.55 | **85.98** | 46.32 | +39.66 |
| CVC-ColonDB (9:1) | 35.30 | **83.09** | 19.86 | +63.23 |
| ETIS (9:1) | 42.32 | **87.10** | 44.36 | +42.74 |
| **All（加权）** | 56.38 | **79.66** | 46.62 | **+33.04** |

---

## 协议 A 结果（Train 池 1450，四集全量，未 matched）

协议设置：

- `polyp_match_support_to_query=False`：所有 query（含 Colon/ETIS）都从 Kvasir+Clinic 训练池（1450）选 support
- 测试集：Kvasir 100 / ClinicDB 62 / ColonDB 380 / ETIS 196（全量）
- 加权 All 权重：100 : 62 : 380 : 196
- random 为原版：整个测试集共用一张固定 support，不是每个 query 独立抽样

| 方法 | All | Kvasir | Clinic | Colon | ETIS | vs random ΔAll |
|------|-----|--------|--------|-------|------|----------------|
| random（原版） | 68.57 | 81.85 | 69.04 | 66.48 | 65.71 | — |
| top1 GAP | 59.69 | 73.89 | 84.93 | 57.29 | 49.11 | −8.88 |
| top1 spatial | 59.12 | 73.65 | 86.88 | 56.83 | 47.39 | −9.45 |
| QSPA gap k3 | 65.66 | 82.10 | 85.81 | 62.16 | 57.70 | −2.91 |
| QSPA gap k5 | 68.57 | 83.33 | 85.21 | 66.33 | 60.11 | ≈0 |
| QSPA spatial k3 | 68.26 | 81.55 | 87.16 | 64.97 | 61.88 | −0.31 |
| **QSPA spatial k5** | **71.26** | 82.41 | 87.32 | 67.97 | 66.87 | **+2.69** |

结论：

1. 只有 **QSPA spatial k5** 在 All 上超过协议 A 的 random（+2.69）。
2. **top1 单独使用（GAP 或 spatial）在协议 A 下下降约 9 个点**，主要因为 Colon/ETIS 的 support 来自 Kvasir+Clinic 的跨池选择，单张 top1 不稳定。
3. QSPA gap k5 与 random 持平（68.57），k3 略低。

注意：

- 协议 A 的 random 是固定单张 support，与协议 B 的逐 query random（All 56.38）不可直接比较。
- 协议 A 与协议 B 的 top1 不能放在同一张 Δ 表中对比。
- 结果目录：`official_benchmark_rerun/polyp_protocol_a/`；日志：`official_benchmark_rerun/polyp_protocol_a_parallel/`（全部 DONE，无 FAIL）。


