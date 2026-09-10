请在当前 ProtoSAM 项目中实现一个“Query-conditioned Support Prototype Aggregation”功能。项目原本是 1-way 1-shot ProtoSAM：给定一张 support image、对应的 support mask，以及一张 query image；通过 DINOv2 提取特征、ALPNet prototype matching 生成 coarse mask，再提取 SAM prompts，最后由 SAM 输出分割结果。

目标：
将原来的“随机选择一张 support image”改为“根据 query image 与候选 support images 的 DINOv2 特征相似度，选择并融合 top-K 个最匹配的 support prototypes”。先只修改 prototype aggregation 部分，不改变 ProtoSAM 后续的 coarse segmentation、prompt extraction 和 SAM inference 流程。

请先阅读当前代码结构，找到以下逻辑：
1. support/query 图像经过 DINOv2 的特征提取代码；
2. 根据 support mask 构造 foreground/background global prototype 和 local prototype 的代码；
3. query feature 与 prototype 计算 cosine similarity 并生成 coarse mask 的代码；
4. 从 coarse mask 提取 bounding box、centroid point、confidence point 的代码；
5. SAM/MedSAM 最终推理代码。

不要重写整个项目。优先进行最小侵入式修改，尽量复用现有 ProtoSAM 的 prototype construction 和 SAM pipeline。

一、输入设定

将 support 输入从单个样本扩展为 support pool：

support_images = [x_s_1, ..., x_s_N]
support_masks  = [y_s_1, ..., y_s_N]

query 输入仍然是一张图像：

query_image = x_q

每张 support image 都有一个像素级 support mask，所有 support mask 表示同一个目标类别。仍然保持 1-way 设定。N 是候选 support pool 的大小，K 是实际用于 prototype aggregation 的 top-K 数量，满足：

1 <= K <= N

如果当前数据加载器或实验设置仍然只能提供一张 support image，也要保证 K=1 时行为与原始 ProtoSAM 尽量一致。

二、计算 DINOv2 image-level feature

对每张 support image 和 query image 使用 ProtoSAM 当前已有的 DINOv2 encoder，不引入新的 encoder：

F_i^s = DINOv2(x_i^s)
F^q   = DINOv2(x^q)

其中 F_i^s 和 F^q 是空间特征图，形状可能是 [C, Hf, Wf]、[B, C, Hf, Wf] 或 token 形式。请根据当前代码的实际张量格式实现，不要假设固定形状。

从空间 feature map 得到用于 support retrieval 的 global feature：

g_i^s = GAP(F_i^s)
g^q   = GAP(F^q)

其中 GAP 是对空间维度进行 global average pooling。如果 DINOv2 输出包含 CLS token，请按照当前项目已有处理方式去除 CLS token 后再恢复空间结构。

在计算 cosine similarity 前，对 global feature 做 L2 normalization：

g_i^s = normalize(g_i^s)
g^q   = normalize(g^q)

三、根据 query-support similarity 选择 top-K support

计算每张 support image 与 query image 的全局 DINOv2 cosine similarity：

r_i = cosine_similarity(g^q, g_i^s)

得到：

r = [r_1, r_2, ..., r_N]

按照 r_i 从高到低排序，选择 top-K support：

N_K(q) = TopK(r, K)

请返回或记录：
- selected support indices；
- selected support similarities；
- 所有 support 的 similarity；
- top-K support 的排序结果。

建议增加 debug/log 输出，方便确认 support selection 是否正确，但不要在默认模式下打印过多内容。

四、计算 similarity-based aggregation weight

对 top-K support 使用 temperature-scaled softmax 计算权重：

w_i =
exp(r_i / T) /
sum_j exp(r_j / T)

其中 i,j 只遍历 top-K support，T 是可配置的 temperature，默认建议设为 0.07 或根据当前项目风格选择一个合理默认值。

要求：
1. 权重必须非负；
2. top-K 权重之和必须接近 1；
3. 使用 torch.softmax 或数值稳定的实现；
4. 不要直接使用 w_i = r_i / sum(r)，因为 cosine similarity 可能为负；
5. 在 batch inference 中支持 batch 维度；
6. 使用 torch.no_grad() 的 inference 路径，不额外训练 DINOv2。

请增加配置参数，例如：

--support_selection random
--support_selection top1
--support_selection topk_weighted

--top_k K
--prototype_temperature T

建议默认模式可以设为 topk_weighted，但保留 random 作为原始 baseline，保留 top1 作为 ablation。

五、每个 support 独立构造 foreground/background prototype

对于每个 support image x_i^s 和对应 mask y_i^s，复用当前 ProtoSAM 的 prototype extraction 代码，分别得到：

P_i^fg
P_i^bg

如果当前实现包含：
- foreground global prototype；
- background global prototype；
- foreground local prototypes；
- background local prototypes；

请保持这些 prototype 的计算逻辑不变，只是在 support 维度上重复执行。

重要：
不要把不同 support 的 foreground 和 background 特征混在一起。必须分别维护：

P_i^fg 和 P_i^bg

不能计算类似：

mean(P_i^fg + P_i^bg)

而应该分别聚合 foreground 和 background。

六、构造 proxy foreground/background prototype

对于 top-K support 的 global prototype，使用 query-conditioned weighted aggregation：

P_proxy^fg(q) =
sum_{i in TopK(q)} w_i P_i^fg

P_proxy^bg(q) =
sum_{i in TopK(q)} w_i P_i^bg

也就是用同一组 support weights 分别聚合 foreground 和 background prototype。

请确保所有 support prototype 位于同一个 DINOv2 feature space 中。如果 prototype 是向量 [C]，按 support 维度进行加权求和；如果当前代码中的 prototype 带有额外 local/prototype 维度，请按照下面的规则处理。

七、local prototype 的处理方式

不同 support image 的 local prototype 编号没有语义对应关系，因此不要简单执行：

P_1^local + P_2^local + ...

优先采用以下实现方式之一，按照当前代码最容易集成的方式选择：

方案 A，推荐：
1. 每个 support 独立使用其 local prototypes 与 query feature map 计算 local similarity map；
2. 对每个 support 得到 foreground local similarity map 和 background local similarity map；
3. 再使用 support-level weight w_i 融合各 support 的 similarity map：

M_proxy,c^local(u,v) =
sum_i w_i M_i,c^local(u,v)

其中 c 属于 {fg, bg}。

方案 B：
将 top-K support 的 local prototypes 合并为 prototype pool；
对 query 每个位置与整个 prototype pool 计算相似度；
使用 top-M average 或 max 进行聚合。

如果当前代码改动复杂，第一版可以只对 global prototype 做 proxy aggregation，并保留 local prototype 的原始 ProtoSAM 逻辑；但请在代码中明确说明采用了哪种方式，并将 global-only 版本作为可运行 baseline。

八、生成 coarse segmentation

在现有 ProtoSAM coarse segmentation 代码中，将原来的单 support prototype 替换为 proxy prototype 或 proxy similarity map。

对于 global prototype，计算：

M_proxy^fg(u,v) =
cos(F^q(u,v), P_proxy^fg(q))

M_proxy^bg(u,v) =
cos(F^q(u,v), P_proxy^bg(q))

如果当前 ProtoSAM 使用缩放因子 alpha，请保持原有 alpha，不要随意修改：

S_proxy,c(u,v) = alpha * M_proxy,c(u,v)

再按照原始 ProtoSAM 的方式对 foreground/background similarity map 做融合和 softmax，生成 coarse mask/probability map。

最终粗分割流程应该仍然是：

query feature
→ foreground/background proxy similarity
→ class-wise score fusion
→ softmax
→ coarse segmentation mask

不要在这一阶段调用 ground-truth query mask。

九、后续 SAM pipeline 保持不变

coarse mask 生成后，继续复用当前 ProtoSAM 的原始流程：

1. connected component analysis；
2. 计算每个 component 的平均 foreground confidence；
3. 选择最高置信度 component；
4. 提取 bounding box；
5. 提取 centroid point；
6. 提取 confidence-based positive point；
7. 将 prompts 输入 SAM 或 MedSAM；
8. 输出最终 segmentation mask。

action
→ 原始 SAM inference