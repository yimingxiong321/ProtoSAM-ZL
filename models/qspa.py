"""Query-conditioned Support Prototype Aggregation (QSPA).

Retrieves top-K supports by DINOv2 GAP cosine similarity and computes
temperature-scaled softmax weights. Prototype / similarity fusion lives in
FewShotSeg (Scheme A: per-support similarity maps, then weighted sum).
"""
import torch
import torch.nn.functional as F


VALID_SUPPORT_SELECTION = ("random", "top1", "topk_weighted")


def resolve_support_selection(config):
    """Map new support_selection and legacy support_select_mode.

    support_selection: random | top1 | topk_weighted
    support_select_mode=dino_sim is treated as top1 when support_selection is unset/random.
    """
    sel = config.get("support_selection", "random")
    legacy = config.get("support_select_mode", "random")
    if sel in ("top1", "topk_weighted"):
        return sel
    if sel == "random" and legacy == "dino_sim":
        return "top1"
    if sel in VALID_SUPPORT_SELECTION:
        return sel
    return "random"


def uses_dino_support_pool(selection):
    return selection in ("top1", "topk_weighted")


def selection_top_k(selection, top_k):
    if selection == "top1":
        return 1
    return max(1, int(top_k))


@torch.no_grad()
def aggregation_weights(similarities, temperature=0.07):
    """Temperature-scaled softmax over top-K cosine similarities.

    similarities: [K] or [B, K]. Cosine can be negative, so do not use r/sum(r).
    """
    sims = similarities
    if not torch.is_tensor(sims):
        sims = torch.as_tensor(sims, dtype=torch.float32)
    if sims.dim() == 0:
        sims = sims.view(1)
    temp = max(float(temperature), 1e-6)
    return torch.softmax(sims / temp, dim=-1)


@torch.no_grad()
def rank_supports_by_dino_sim(encoder, query_images, support_embs, pool_indices=None):
    """Return cosine similarities and descending global pool rank.

    query_images: [B, 3, H, W] (B=1 in current eval).
    support_embs: [N, C] L2-normalized DINOv2 GAP features.
    pool_indices: optional global indices restricting retrieval to a sub-pool.
    """
    q_emb = encoder.get_image_embedding(query_images)
    if q_emb.dim() == 1:
        q_emb = q_emb.unsqueeze(0)
    if q_emb.shape[0] > 1:
        q_emb = F.normalize(q_emb.mean(dim=0, keepdim=True), p=2, dim=1, eps=1e-4)

    candidate_embs = support_embs.to(q_emb.device)
    global_indices = None
    if pool_indices is not None:
        if len(pool_indices) == 0:
            raise ValueError("rank_supports_by_dino_sim received empty pool_indices")
        global_indices = torch.as_tensor(pool_indices, dtype=torch.long, device=candidate_embs.device)
        candidate_embs = candidate_embs.index_select(0, global_indices)

    sims = F.cosine_similarity(q_emb, candidate_embs, dim=1)
    local_order = torch.argsort(sims, descending=True)
    ranked_sims = sims.index_select(0, local_order)
    if global_indices is not None:
        order = global_indices.index_select(0, local_order)
    else:
        order = local_order
    return ranked_sims, order


@torch.no_grad()
def select_topk_supports_by_dino_sim(
    encoder,
    query_images,
    support_embs,
    pool_paths,
    dataset,
    top_k=1,
    temperature=0.07,
    pool_indices=None,
    query_dataset=None,
):
    """Load top-K supports and QSPA softmax weights.

    Returns images, masks, cases, weights [K], info dict.
    pool_indices restricts retrieval; query_dataset is recorded for logging.
    """
    ranked_sims, order = rank_supports_by_dino_sim(
        encoder, query_images, support_embs, pool_indices=pool_indices)
    k = min(int(top_k), int(ranked_sims.numel()))
    top_idx = order[:k]
    top_sims = ranked_sims[:k]
    weights = aggregation_weights(top_sims, temperature=temperature)

    images, masks, cases = [], [], []
    for idx in top_idx.tolist():
        img, mask, case = dataset.load_support_item(*pool_paths[idx])
        images.append(img)
        masks.append(mask)
        cases.append(case)

    info = {
        "selected_indices": top_idx.tolist(),
        "selected_similarities": [float(x) for x in top_sims.detach().cpu()],
        "weights": [float(x) for x in weights.detach().cpu()],
        "ranked": [
            (int(i), float(s), float(w))
            for i, s, w in zip(top_idx.tolist(), top_sims.tolist(), weights.tolist())
        ],
        "n_pool": int(len(pool_indices) if pool_indices is not None else ranked_sims.numel()),
        "top_k": k,
        "temperature": float(temperature),
        "query_dataset": query_dataset,
        "pool_indices": list(pool_indices) if pool_indices is not None else None,
        "all_similarities": ranked_sims.detach().cpu(),
    }
    return images, masks, cases, weights, info
