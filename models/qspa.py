"""Query-conditioned Support Prototype Aggregation (QSPA).

Retrieves top-K supports by DINOv2 GAP cosine similarity and computes
temperature-scaled softmax weights. Prototype / similarity fusion lives in
FewShotSeg (Scheme A: per-support similarity maps, then weighted sum).
"""
import torch
import torch.nn.functional as F


VALID_SUPPORT_SELECTION = ("random", "top1", "topk_weighted")
VALID_RETRIEVAL_MODES = ("gap", "spatial")


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


def uses_qspa_weighted_fusion(selection):
    """True only for multi-support softmax fusion (Scheme A). top1 is retrieval-only."""
    return selection == "topk_weighted"


def coarse_support_weights(selection, weights, top_k_effective):
    """Weights passed to FewShotSeg; None keeps legacy single-support forward."""
    if not uses_qspa_weighted_fusion(selection):
        return None
    if int(top_k_effective) <= 1:
        return None
    return weights


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
def spatial_retrieval_similarity(q_fts, s_fts):
    """Image-level score from aligned DINO spatial feature maps.

    Per spatial location, L2-normalize channel vectors and take cosine similarity,
    then average over H x W. q_fts: [1,C,H,W], s_fts: [N,C,H,W].
    """
    if q_fts.dim() == 3:
        q_fts = q_fts.unsqueeze(0)
    if q_fts.shape[0] > 1:
        q_fts = q_fts.mean(dim=0, keepdim=True)
    q = F.normalize(q_fts.float(), p=2, dim=1, eps=1e-4)
    s = F.normalize(s_fts.float(), p=2, dim=1, eps=1e-4)
    return (q * s).sum(dim=1).mean(dim=(-2, -1))


@torch.no_grad()
def rank_supports_by_gap_sim(encoder, query_images, support_embs, pool_indices=None):
    """Return GAP cosine similarities and descending global pool rank."""
    q_emb = encoder.get_image_embedding(query_images)
    if q_emb.dim() == 1:
        q_emb = q_emb.unsqueeze(0)
    if q_emb.shape[0] > 1:
        q_emb = F.normalize(q_emb.mean(dim=0, keepdim=True), p=2, dim=1, eps=1e-4)

    candidate_embs = support_embs.to(q_emb.device)
    global_indices = None
    if pool_indices is not None:
        if len(pool_indices) == 0:
            raise ValueError("rank_supports_by_gap_sim received empty pool_indices")
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
def rank_supports_by_spatial_sim(
    encoder,
    query_images,
    support_spatial_fts,
    pool_indices=None,
    chunk_size=64,
):
    """Return spatial-map similarities and descending global pool rank."""
    q_fts = encoder.get_features(query_images)
    device = q_fts.device
    global_indices = None
    candidates = support_spatial_fts
    if pool_indices is not None:
        if len(pool_indices) == 0:
            raise ValueError("rank_supports_by_spatial_sim received empty pool_indices")
        global_indices = torch.as_tensor(pool_indices, dtype=torch.long)
        candidates = support_spatial_fts.index_select(0, global_indices)

    sims_list = []
    for start in range(0, candidates.shape[0], chunk_size):
        chunk = candidates[start:start + chunk_size].to(device)
        sims_list.append(spatial_retrieval_similarity(q_fts, chunk).detach().cpu())
    sims = torch.cat(sims_list)
    local_order = torch.argsort(sims, descending=True)
    ranked_sims = sims.index_select(0, local_order)
    if global_indices is not None:
        order = global_indices.index_select(0, local_order)
    else:
        order = local_order
    return ranked_sims, order


@torch.no_grad()
def rank_supports(
    encoder,
    query_images,
    support_features,
    retrieval_mode="gap",
    pool_indices=None,
    spatial_chunk_size=64,
):
    """Rank support pool by GAP or spatial DINO feature similarity."""
    if retrieval_mode not in VALID_RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode must be one of {VALID_RETRIEVAL_MODES}, got {retrieval_mode!r}")
    if retrieval_mode == "spatial":
        return rank_supports_by_spatial_sim(
            encoder, query_images, support_features,
            pool_indices=pool_indices, chunk_size=spatial_chunk_size)
    return rank_supports_by_gap_sim(
        encoder, query_images, support_features, pool_indices=pool_indices)


# Backward-compatible alias
rank_supports_by_dino_sim = rank_supports_by_gap_sim


@torch.no_grad()
def select_top1_supports_by_dino_sim(
    encoder,
    query_images,
    support_features,
    pool_paths=None,
    dataset=None,
    pool_indices=None,
    query_dataset=None,
    retrieval_mode="gap",
    spatial_chunk_size=64,
    load_support_fn=None,
):
    """DINO top-1 retrieval (no temperature aggregation). Same rank/load as top-K with k=1."""
    return select_topk_supports_by_dino_sim(
        encoder,
        query_images,
        support_features,
        pool_paths=pool_paths,
        dataset=dataset,
        top_k=1,
        temperature=1.0,
        pool_indices=pool_indices,
        query_dataset=query_dataset,
        retrieval_mode=retrieval_mode,
        spatial_chunk_size=spatial_chunk_size,
        load_support_fn=load_support_fn,
    )


@torch.no_grad()
def select_supports_by_dino_sim(
    encoder,
    query_images,
    support_features,
    support_selection="topk_weighted",
    top_k=5,
    temperature=0.07,
    pool_paths=None,
    dataset=None,
    pool_indices=None,
    query_dataset=None,
    retrieval_mode="gap",
    spatial_chunk_size=64,
    load_support_fn=None,
):
    """Functional entry: top1 | topk_weighted | random (random must be handled by caller)."""
    selection = resolve_support_selection({"support_selection": support_selection})
    if selection == "top1":
        return select_top1_supports_by_dino_sim(
            encoder,
            query_images,
            support_features,
            pool_paths=pool_paths,
            dataset=dataset,
            pool_indices=pool_indices,
            query_dataset=query_dataset,
            retrieval_mode=retrieval_mode,
            spatial_chunk_size=spatial_chunk_size,
            load_support_fn=load_support_fn,
        )
    if selection == "topk_weighted":
        k = selection_top_k(selection, top_k)
        return select_topk_supports_by_dino_sim(
            encoder,
            query_images,
            support_features,
            pool_paths=pool_paths,
            dataset=dataset,
            top_k=k,
            temperature=temperature,
            pool_indices=pool_indices,
            query_dataset=query_dataset,
            retrieval_mode=retrieval_mode,
            spatial_chunk_size=spatial_chunk_size,
            load_support_fn=load_support_fn,
        )
    raise ValueError(
        f"select_supports_by_dino_sim does not handle support_selection={support_selection!r}; "
        "use random support sampling in the dataset loop."
    )


@torch.no_grad()
def select_topk_supports_by_dino_sim(
    encoder,
    query_images,
    support_features,
    pool_paths=None,
    dataset=None,
    top_k=1,
    temperature=0.07,
    pool_indices=None,
    query_dataset=None,
    retrieval_mode="gap",
    spatial_chunk_size=64,
    load_support_fn=None,
):
    """Load top-K supports and QSPA softmax weights.

    Returns images, masks, cases, weights [K], info dict.
    pool_indices restricts retrieval; query_dataset is recorded for logging.
    support_features: [N,C] for gap or [N,C,H,W] for spatial retrieval.
    load_support_fn: optional callable(pool_index) -> (image, mask, case).
    """
    ranked_sims, order = rank_supports(
        encoder, query_images, support_features,
        retrieval_mode=retrieval_mode,
        pool_indices=pool_indices,
        spatial_chunk_size=spatial_chunk_size,
    )
    k = min(int(top_k), int(ranked_sims.numel()))
    top_idx = order[:k]
    top_sims = ranked_sims[:k]
    weights = aggregation_weights(top_sims, temperature=temperature)

    images, masks, cases = [], [], []
    for idx in top_idx.tolist():
        if load_support_fn is not None:
            img, mask, case = load_support_fn(int(idx))
        elif pool_paths is not None and dataset is not None:
            img, mask, case = dataset.load_support_item(*pool_paths[idx])
        else:
            raise ValueError(
                "select_topk_supports requires load_support_fn or pool_paths+dataset")
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
        "retrieval_mode": retrieval_mode,
        "pool_indices": list(pool_indices) if pool_indices is not None else None,
        "all_similarities": ranked_sims.detach().cpu(),
    }
    return images, masks, cases, weights, info
