"""
Validation script
"""
import math
import os
import hashlib
import pandas as pd
import csv
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import time
import matplotlib.pyplot as plt
from models.ProtoSAM import ProtoSAM,  ALPNetWrapper, SamWrapperWrapper, InputFactory, ModelWrapper, TYPE_ALPNET, TYPE_SAM
from models.ProtoMedSAM import ProtoMedSAM
from models.grid_proto_fewshot import FewShotSeg
from models.qspa import (
    resolve_support_selection,
    uses_dino_support_pool,
    selection_top_k,
    coarse_support_weights,
    select_supports_by_dino_sim,
    select_top1_supports_by_dino_sim,
)
from models.polyp_support import (
    build_support_pool_by_dataset,
    pool_summary,
    resolve_pool_indices,
    resolve_query_dataset,
    select_random_matched_support,
)
from models.alp_support import (
    build_alp_support_pool,
    make_alp_support_loader,
    resolve_alp_support_scan_ids,
)
from models.segment_anything.utils.transforms import ResizeLongestSide
from models.SamWrapper import SamWrapper
# from dataloaders.PolypDataset import get_polyp_dataset, get_vps_easy_unseen_dataset, get_vps_hard_unseen_dataset, PolypDataset, KVASIR, CVC300, COLON_DB, ETIS_DB, CLINIC_DB
from dataloaders.PolypDataset import get_polyp_dataset, PolypDataset
from dataloaders.PolypTransforms import get_polyp_transform
from dataloaders.SimpleDataset import SimpleDataset
from dataloaders.ManualAnnoDatasetv2 import get_nii_dataset
from dataloaders.common import ValidationDataset
from config_ssl_upload import ex

import tqdm
from tqdm.auto import tqdm
import cv2
from collections import defaultdict
from util.presence_metrics import (
    is_finite_number,
    mask_presence_metrics,
    ranking_metrics,
)
from util.candidate_audit import summarize_proposal_slices

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

# Supported Datasets
CHAOS = "chaos"
SABS = "sabs"
POLYPS = "polyps"

ALP_DS = [CHAOS, SABS]

ROT_DEG = 0

PRESENCE_SCORE_NAMES = (
    "max_probability",
    "topk_mean_probability",
    "predicted_area_ratio",
    "max_fg_bg_likelihood_ratio",
    "fg_bg_likelihood_ratio",
    "probability_srqs_strength",
    "probability_srqs_compactness",
    "probability_srqs_purity",
    "probability_srqs_score",
    "likelihood_srqs_strength",
    "likelihood_srqs_compactness",
    "likelihood_srqs_purity",
    "likelihood_srqs_score",
    "patchcore_candidate_mean_similarity",
    "patchcore_candidate_worst_similarity",
    "patchcore_support_coverage",
    "patchcore_bidirectional_score",
)


def _plain_sample_value(sample, key, default=""):
    value = sample.get(key, default)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    if torch.is_tensor(value):
        value = value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


def annotate_presence_slice_distance(rows):
    """Mark near-boundary hard negatives using only slices in each volume."""
    volumes = defaultdict(list)
    for row in rows:
        row["distance_to_positive_slice"] = ""
        row["hard_negative_3"] = False
        row["hard_negative_5"] = False
        scan_id = str(row.get("scan_id", ""))
        z_id = row.get("z_id")
        if scan_id and is_finite_number(z_id):
            volumes[scan_id].append(row)

    for volume_rows in volumes.values():
        positive_z = [float(row["z_id"]) for row in volume_rows if row["gt_present"]]
        for row in volume_rows:
            if row["gt_present"] or not positive_z:
                row["distance_to_positive_slice"] = 0.0 if row["gt_present"] else ""
                row["hard_negative_3"] = False
                row["hard_negative_5"] = False
                continue
            distance = min(abs(float(row["z_id"]) - z) for z in positive_z)
            row["distance_to_positive_slice"] = distance
            row["hard_negative_3"] = distance <= 3
            row["hard_negative_5"] = distance <= 5


def _safe_mean(values):
    return float(np.mean(values)) if values else float("nan")

def get_bounding_box(segmentation_map):
    """Generate bounding box from a segmentation map. one bounding box to include the extreme points of the segmentation map."""
    if isinstance(segmentation_map, torch.Tensor):
        segmentation_map = segmentation_map.cpu().numpy()
    
    bbox = cv2.boundingRect(segmentation_map.astype(np.uint8))
    # plot bounding boxes for each contours
    # plt.figure()
    # x, y, w, h = bbox
    # plt.imshow(segmentation_map)
    # plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='r', linewidth=2))
    # plt.savefig("debug/bounding_boxes.png") 

    return bbox

def calc_iou(boxA, boxB):
    """
    boxA: [x, y, w, h]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def eval_detection(pred_list):
    """
    pred_list: list of dictionaries with keys 'pred_bbox', 'gt_bbox' and score (prediction confidence score).
    compute AP50, AP75, AP50:95:10
    """
    iou_thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)
    ap_dict = {iou: [] for iou in iou_thresholds}
    for iou_threshold in iou_thresholds:
        tp, fp = 0, 0
        
        for pred in pred_list:
            pred_bbox = pred['pred_bbox']
            gt_bbox = pred['gt_bbox']
            
            iou = calc_iou(pred_bbox, gt_bbox)
            
            if iou >= iou_threshold:
                tp += 1
            else:
                fp += 1

        precision = tp / (tp + fp)
        recall = tp / len(pred_list) 
        f1 = 2 * (precision * recall) / (precision + recall)        

        ap_dict[iou_threshold] = {
            'iou_threshold': iou_threshold,
            'tp': tp,
            'fp': fp,
            'n_gt': len(pred_list),
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Convert results to a DataFrame and save to CSV
    results = []
    for iou_threshold in iou_thresholds:
        results.append(ap_dict[iou_threshold])
    
    df = pd.DataFrame(results)
    return df


def plot_pred_gt_support(query_image, pred, gt, support_images, support_masks, score=None, save_path="debug/pred_vs_gt.png"):
    """
    pred: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    gt: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    support: 4d tensor of shape (N, C, H, W) where 1 represents foreground and 0 represents background
    """
    if support_images:
        if isinstance(support_images, list):
            support_images = torch.cat(support_images, dim=0).clone().detach()
        if isinstance(support_masks, list):
            support_masks = torch.cat(support_masks, dim=0).clone().detach()
        if len(query_image.shape) == 3:
            query_image = query_image.permute(1, 2, 0).clone().detach()
        if len(support_images.shape) == 4:
            support_images = support_images.clone().detach().permute(0, 2, 3, 1)
        n_support_rows = math.ceil(support_images.shape[0] / 2)
    else:
        n_support_rows = 1
    fig, ax = plt.subplots(n_support_rows + 1, 2)
    query_image = (query_image - query_image.min()) / (query_image.max() - query_image.min())
    ax[0, 0].imshow(query_image.cpu().detach())
    ax[0, 0].imshow(pred, alpha=0.5)
    ax[0, 0].set_title("pred")
    ax[0, 1].imshow(query_image.cpu().detach())
    ax[0, 1].imshow(gt, alpha=0.5)
    ax[0, 1].set_title("gt")
    if support_images is not None:
        for i in range(1, n_support_rows + 1):
            support_images[(i - 1) * 2] = (support_images[(i - 1) * 2] - support_images[(i - 1) * 2].min()) / (support_images[(i - 1) * 2].max() - support_images[(i - 1) * 2].min())
            ax[i, 0].imshow(support_images[(i - 1) * 2].cpu().detach())
            ax[i, 0].imshow(support_masks[(i - 1) * 2].cpu(), alpha=0.5)
            ax[i, 0].set_title(f"support")
            if (i - 1) * 2 + 1 < support_images.shape[0]:
                support_images[(i - 1) * 2 + 1] = (support_images[(i - 1) * 2 + 1] - support_images[(i - 1) * 2 + 1].min()) / (support_images[(i - 1) * 2 + 1].max() - support_images[(i - 1) * 2 + 1].min())
                ax[i, 1].imshow(support_images[(i - 1) * 2 + 1].cpu().detach())
                ax[i, 1].imshow(support_masks[(i - 1) * 2 + 1].cpu(), alpha=0.5)
                ax[i, 1].set_title(f"support")
    if score is not None:
        # plt.title(f"score: {score}") 
        fig.suptitle(f"sam score: {score}")
    fig.savefig(save_path)
    plt.close(fig)


def get_dice_iou_precision_recall(pred: torch.Tensor, gt: torch.Tensor):
    """
    pred: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    gt: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    """
    if gt.sum() == 0:
        print("gt is all background")
        return {"dice": 0, "iou": 0, "precision": 0, "recall": 0}

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def get_boundary_metrics(pred: torch.Tensor, gt: torch.Tensor, tolerance_px: float = 2.0):
    """Surface Dice, GT-boundary recall, and symmetric HD95 in image pixels."""
    pred_np = np.asarray(pred.detach().cpu().squeeze() > 0, dtype=np.uint8)
    gt_np = np.asarray(gt.detach().cpu().squeeze() > 0, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_boundary = pred_np.astype(bool) & ~cv2.erode(pred_np, kernel, iterations=1).astype(bool)
    gt_boundary = gt_np.astype(bool) & ~cv2.erode(gt_np, kernel, iterations=1).astype(bool)

    n_pred = int(pred_boundary.sum())
    n_gt = int(gt_boundary.sum())
    if n_pred == 0 and n_gt == 0:
        return {"boundary_dice": 1.0, "boundary_recall": 1.0, "hd95": 0.0}
    if n_pred == 0 or n_gt == 0:
        diagonal = float(math.hypot(*pred_np.shape))
        return {
            "boundary_dice": 0.0,
            "boundary_recall": 0.0 if n_gt else 1.0,
            "hd95": diagonal,
        }

    # distanceTransform computes distance to zero pixels, so boundaries are zero.
    dist_to_gt = cv2.distanceTransform((~gt_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_pred = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    pred_to_gt = dist_to_gt[pred_boundary]
    gt_to_pred = dist_to_pred[gt_boundary]

    boundary_precision = float(np.mean(pred_to_gt <= tolerance_px))
    boundary_recall = float(np.mean(gt_to_pred <= tolerance_px))
    boundary_dice = (
        2.0 * boundary_precision * boundary_recall
        / (boundary_precision + boundary_recall + 1e-8)
    )
    hd95 = float(np.percentile(np.concatenate([pred_to_gt, gt_to_pred]), 95))
    return {
        "boundary_dice": boundary_dice,
        "boundary_recall": boundary_recall,
        "hd95": hd95,
    }


def get_alpnet_model(_config) -> ModelWrapper:
    alpnet = FewShotSeg(
       _config["input_size"][0],
       _config["reload_model_path"],
       _config["model"]
    )
    alpnet.cuda()
    alpnet_wrapper = ALPNetWrapper(alpnet)
    
    return alpnet_wrapper

def get_sam_model(_config) -> ModelWrapper:
    sam_args = {
        "model_type": "vit_h",
        "sam_checkpoint": "pretrained_model/sam_vit_h.pth"
    }
    sam = SamWrapper(sam_args=sam_args).cuda()
    sam_wrapper = SamWrapperWrapper(sam)
    return sam_wrapper  

def get_model(_config) -> ProtoSAM:
    # Initial Segmentation Model
    if _config["base_model"] == TYPE_ALPNET:
        base_model = get_alpnet_model(_config)
    else:
        raise NotImplementedError(f"base model {_config['base_model']} not implemented")
    
    # ProtoSAM model
    if _config["protosam_sam_ver"] in ("sam_h", "sam_b", "sam3"):
        sam_h_checkpoint = "pretrained_model/sam_vit_h.pth"
        sam_b_checkpoint = "pretrained_model/sam_vit_b.pth"
        default_sam3 = "/share/home/huafuchen01/huangwei/WangRuiFeng/MedicalSAM3/checkpoint/sam3.pt"
        if _config["protosam_sam_ver"] == "sam3":
            sam_checkpoint = _config.get("sam3_checkpoint") or default_sam3
        elif _config["protosam_sam_ver"] == "sam_h":
            sam_checkpoint = sam_h_checkpoint
        else:
            sam_checkpoint = sam_b_checkpoint
        model = ProtoSAM(image_size = (1024, 1024),
                    coarse_segmentation_model=base_model,
                    use_bbox=_config["use_bbox"],
                    use_points=_config["use_points"],
                    use_mask=_config["use_mask"],
                    debug=_config["debug"],
                    num_points_for_sam=1,
                    use_cca=_config["do_cca"],
                    point_mode=_config["point_mode"],
                    use_sam_trans=True, 
                    coarse_pred_only=_config["coarse_pred_only"],
                    sam_pretrained_path=sam_checkpoint,
                    use_neg_points=_config["use_neg_points"],
                    ablation_mode=_config.get("ablation_mode", "none"),
                    ablation_fixes=_config.get("ablation_fixes", None),
                    candidate_audit=_config.get("candidate_audit", False),
                    candidate_audit_thresholds=_config.get(
                        "candidate_audit_thresholds", (0.3, 0.4, 0.5)
                    ),) 
    elif _config["protosam_sam_ver"] == "medsam":
        model = ProtoMedSAM(image_size = (1024, 1024),
                            coarse_segmentation_model=base_model,
                            debug=_config["debug"],
                            use_cca=_config["do_cca"],
        )
    else:
        raise NotImplementedError(f"protosam_sam_ver {_config['protosam_sam_ver']} not implemented")
    
    return model


def get_fewshot_seg(model) -> FewShotSeg:
    coarse = model.coarse_segmentation_model
    if hasattr(coarse, "model"):
        return coarse.model
    return coarse


def load_polyp_path_pairs(txt_path: str):
    pairs = []
    with open(txt_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            image_path, mask_path = line.split()
            pairs.append((image_path, mask_path))
    if not pairs:
        raise ValueError(f"empty polyp split file: {txt_path}")
    return pairs


def apply_polyp_colon_etis_split(te_dataset: PolypDataset, split_dir: str, eval_datasets):
    """Replace test set with 10% query split; return support path pairs for 90% pool."""
    if isinstance(eval_datasets, str):
        eval_datasets = [eval_datasets]
    if not eval_datasets or len(eval_datasets) != 1:
        raise ValueError(
            "polyp_colon_etis_split_dir requires polyp_eval_datasets with exactly one dataset"
        )
    ds_name = eval_datasets[0]
    split_dir = os.path.abspath(split_dir)
    support_txt = os.path.join(split_dir, f"{ds_name}_support.txt")
    test_txt = os.path.join(split_dir, f"{ds_name}_test.txt")
    support_pairs = load_polyp_path_pairs(support_txt)
    test_pairs = load_polyp_path_pairs(test_txt)
    te_dataset.images = [p[0] for p in test_pairs]
    te_dataset.gts = [p[1] for p in test_pairs]
    te_dataset.size = len(test_pairs)
    te_dataset.filter_files_and_get_ds_mean_and_std()
    return support_pairs


def get_support_set_polyps(_config, dataset:PolypDataset):
    n_support = _config["n_support"]
    text_file = _config.get("support_txt_file", None)
    (support_images, support_labels, case) = dataset.get_support(
        n_support=n_support, text_file=text_file)
    
    return support_images, support_labels, case


@torch.no_grad()
def precompute_polyp_support_embeddings(encoder: FewShotSeg, dataset: PolypDataset, pool_paths, batch_size=4, device="cuda"):
    """Cache L2-normalized DINO GAP embeddings for the polyp support pool."""
    embs = []
    for i in tqdm(range(0, len(pool_paths), batch_size), desc="support DINO GAP embeddings"):
        batch_imgs = []
        for image_path, gt_path in pool_paths[i:i + batch_size]:
            img, _, _ = dataset.load_support_item(image_path, gt_path)
            batch_imgs.append(img)
        x = torch.cat(batch_imgs, dim=0).to(device)
        embs.append(encoder.get_image_embedding(x).cpu())
        del x
    return torch.cat(embs, dim=0)


@torch.no_grad()
def precompute_polyp_support_spatial_features(
    encoder: FewShotSeg, dataset: PolypDataset, pool_paths, batch_size=2, device="cuda"):
    """Cache DINO spatial feature maps for the polyp support pool (CPU float16)."""
    feats = []
    for i in tqdm(range(0, len(pool_paths), batch_size), desc="support DINO spatial features"):
        batch_imgs = []
        for image_path, gt_path in pool_paths[i:i + batch_size]:
            img, _, _ = dataset.load_support_item(image_path, gt_path)
            batch_imgs.append(img)
        x = torch.cat(batch_imgs, dim=0).to(device)
        feats.append(encoder.get_features(x).cpu().half())
        del x
    return torch.cat(feats, dim=0)


@torch.no_grad()
def precompute_polyp_support_features(
    encoder: FewShotSeg,
    dataset: PolypDataset,
    pool_paths,
    retrieval_mode="gap",
    batch_size=4,
    device="cuda",
):
    if retrieval_mode == "spatial":
        return precompute_polyp_support_spatial_features(
            encoder, dataset, pool_paths, batch_size=max(1, batch_size // 2), device=device)
    return precompute_polyp_support_embeddings(
        encoder, dataset, pool_paths, batch_size=batch_size, device=device)


@torch.no_grad()
def precompute_alp_support_features(
    encoder: FewShotSeg,
    load_support_fn,
    pool_size: int,
    retrieval_mode="spatial",
    batch_size=4,
    device="cuda",
):
    """Cache DINO features for an ALP support-slice pool."""
    feats = []
    desc = (
        "support DINO GAP embeddings"
        if retrieval_mode == "gap"
        else "support DINO spatial features"
    )
    step = max(1, batch_size // 2) if retrieval_mode == "spatial" else batch_size
    for start in tqdm(range(0, pool_size, step), desc=desc):
        batch_imgs = []
        for pool_idx in range(start, min(start + step, pool_size)):
            img, _, _ = load_support_fn(pool_idx)
            batch_imgs.append(img)
        x = torch.cat(batch_imgs, dim=0).to(device)
        if retrieval_mode == "spatial":
            feats.append(encoder.get_features(x).cpu().half())
        else:
            feats.append(encoder.get_image_embedding(x).cpu())
        del x
    return torch.cat(feats, dim=0)


@torch.no_grad()
def select_polyp_support_by_dino_sim(encoder, query_images, support_embs, pool_paths, dataset: PolypDataset):
    """Legacy top-1 wrapper. Prefer select_top1_supports_by_dino_sim."""
    images, masks, cases, weights, info = select_top1_supports_by_dino_sim(
        encoder, query_images, support_embs, pool_paths=pool_paths, dataset=dataset)
    best = info["selected_indices"][0]
    return images, masks, cases[0], best, info["selected_similarities"][0]


def get_support_set_alpds(config, dataset:ValidationDataset):
    support_set = dataset.get_support_set(config)
    support_fg_masks = support_set["support_labels"]
    support_images = support_set["support_images"]
    support_scan_id = support_set["support_scan_id"]
    return support_images, support_fg_masks, support_scan_id


def get_support_set(_config, dataset):
    if POLYPS in _config["dataset"].lower():
        support_images, support_fg_masks, case = get_support_set_polyps(_config, dataset)
    elif any(item in _config["dataset"].lower() for item in ALP_DS):
        support_images, support_fg_masks, support_scan_id = get_support_set_alpds(_config, dataset)
    else:
        raise NotImplementedError(f"dataset {_config['dataset']} not implemented")
    return support_images, support_fg_masks, support_scan_id


def update_support_set_by_scan_part(support_images, support_labels, qpart):
    qpart_support_images = [support_images[qpart]]
    qpart_support_labels = [support_labels[qpart]]
    
    return qpart_support_images, qpart_support_labels


def manage_support_sets(sample_batched, all_support_images, all_support_fg_mask, support_images, support_fg_mask, qpart=None):
    if sample_batched['part_assign'][0] != qpart:
        qpart = sample_batched['part_assign'][0]
        support_images, support_fg_mask = update_support_set_by_scan_part(all_support_images, all_support_fg_mask, qpart)
            
    return support_images, support_fg_mask, qpart


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        print(f"####### created dir:{_run.observers[0].dir} #######")
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')
    support_selection = resolve_support_selection(_config)
    qspa_top_k = selection_top_k(support_selection, _config.get("top_k", 5))
    qspa_temp = float(_config.get("prototype_temperature", 0.07))
    support_retrieval_mode = _config.get("support_retrieval_mode", "gap")
    spatial_chunk_size = int(_config.get("support_spatial_chunk_size", 64))
    polyp_match_support = bool(_config.get("polyp_match_support_to_query", False))
    polyp_unmatched_policy = _config.get("polyp_unmatched_support_policy", "skip")
    print(
        f"config do_cca: {_config['do_cca']}, use_bbox: {_config['use_bbox']}, "
        f"support_select_mode: {_config.get('support_select_mode', 'random')}, "
        f"support_selection: {support_selection}, top_k: {qspa_top_k}, T: {qspa_temp}, "
        f"support_retrieval_mode: {support_retrieval_mode}, "
        f"polyp_match_support_to_query: {polyp_match_support}, "
        f"polyp_unmatched_support_policy: {polyp_unmatched_policy}"
    )
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    model = get_model(_config)
    model = model.to(torch.device("cuda"))
    model.eval()
    if hasattr(model, "coarse_segmentation_model"):
        model.coarse_segmentation_model.eval()
    
    sam_trans = ResizeLongestSide(1024)
    if POLYPS in _config["dataset"].lower():
        tr_dataset, te_dataset = get_polyp_dataset(sam_trans=sam_trans, image_size=(1024, 1024))
        eval_ds = _config.get("polyp_eval_datasets", None)
        colon_etis_support_pairs = None
        split_dir = _config.get("polyp_colon_etis_split_dir")
        if split_dir:
            eval_ds = _config.get("polyp_eval_datasets")
            colon_etis_support_pairs = apply_polyp_colon_etis_split(
                te_dataset, split_dir, eval_ds)
            _config["_colon_etis_support_pairs"] = colon_etis_support_pairs
            _log.info(
                f'Polyp 9:1 split from {split_dir}: '
                f'test={te_dataset.size} support={len(colon_etis_support_pairs)}'
            )
        elif eval_ds:
            if isinstance(eval_ds, str):
                eval_ds = [eval_ds]
            keep_img, keep_gt = [], []
            for img, gt in zip(te_dataset.images, te_dataset.gts):
                if any(name in img for name in eval_ds):
                    keep_img.append(img)
                    keep_gt.append(gt)
            te_dataset.images = keep_img
            te_dataset.gts = keep_gt
            te_dataset.size = len(keep_img)
            _log.info(f'Polyp eval subsets {list(eval_ds)}: {te_dataset.size} images')
        else:
            colon_etis_support_pairs = None
    elif CHAOS in _config["dataset"].lower() or SABS in _config["dataset"].lower():
        tr_dataset, te_dataset = get_nii_dataset(_config, _config["input_size"][0]) 
    else:
        raise NotImplementedError(
            f"dataset {_config['dataset']} not implemented")

    # dataloaders
    testloader = DataLoader(
        te_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )

    _log.info('###### Starting validation ######')
    model.eval()

    mean_dice = []
    mean_prec = []
    mean_rec = []
    mean_iou = []
    mean_boundary_dice = []
    mean_boundary_rec = []
    mean_hd95 = []
    presence_rows = []
    candidate_rows = []
    proposal_slice_rows = []
    prediction_rows = []
    
    mean_dice_cases = {}
    mean_iou_cases = {} 
    bboxes_w_scores = []
    
    curr_case = None
    supp_fts = None
    qpart = None
    support_images = support_fg_mask = None
    all_support_images, all_support_fg_mask, support_scan_id = None, None, None
    MAX_SUPPORT_IMAGES = 1
    is_alp_ds = any(item in _config["dataset"].lower() for item in ALP_DS)
    is_polyp_ds  = POLYPS in _config["dataset"].lower()
    support_select_mode = _config.get("support_select_mode", "random")
    polyp_pool_paths = None
    polyp_pool_by_dataset = None
    polyp_support_features = None
    alp_load_support_fn = None
    fewshot_seg = None
    qspa_weights = None
    qspa_info = None
    polyp_support_skipped = 0
    
    if is_alp_ds and uses_dino_support_pool(support_selection):
        manual_dataset = te_dataset.dataset
        curr_class = te_dataset.get_curr_cls()
        class_idx = [curr_class]
        support_scan_id = resolve_alp_support_scan_ids(_config, manual_dataset)
        alp_support_pool = build_alp_support_pool(
            manual_dataset, curr_class, support_scan_id)
        alp_load_support_fn = make_alp_support_loader(
            manual_dataset, alp_support_pool, curr_class, class_idx)
        _log.info(
            f'ALP QSPA support pool size: {len(alp_support_pool)} '
            f'scans={support_scan_id} selection={support_selection} '
            f'top_k={qspa_top_k} retrieval={support_retrieval_mode}'
        )
        fewshot_seg = get_fewshot_seg(model)
        fewshot_seg.eval()
        polyp_support_features = precompute_alp_support_features(
            fewshot_seg, alp_load_support_fn, len(alp_support_pool),
            retrieval_mode=support_retrieval_mode)
        if support_retrieval_mode == "gap":
            polyp_support_features = polyp_support_features.cuda()
    elif is_alp_ds:
        all_support_images, all_support_fg_mask, support_scan_id = get_support_set(_config, te_dataset)
    elif is_polyp_ds and (uses_dino_support_pool(support_selection) or polyp_match_support):
        colon_etis_support_pairs = _config.get("_colon_etis_support_pairs")
        if colon_etis_support_pairs is not None:
            polyp_pool_paths = list(colon_etis_support_pairs)
        else:
            polyp_pool_paths = tr_dataset.get_support_pool_paths(
                text_file=_config.get("support_txt_file", None)
            )
        polyp_pool_by_dataset = build_support_pool_by_dataset(polyp_pool_paths)
        _log.info(
            f'Polyp support pool size: {len(polyp_pool_paths)} '
            f'by_dataset={pool_summary(polyp_pool_by_dataset)} '
            f'match_query={polyp_match_support} selection={support_selection} '
            f'top_k={qspa_top_k} retrieval={support_retrieval_mode}'
        )
        if uses_dino_support_pool(support_selection):
            fewshot_seg = get_fewshot_seg(model)
            fewshot_seg.eval()
            polyp_support_features = precompute_polyp_support_features(
                fewshot_seg, tr_dataset, polyp_pool_paths,
                retrieval_mode=support_retrieval_mode)
            if support_retrieval_mode == "gap":
                polyp_support_features = polyp_support_features.cuda()
    elif is_polyp_ds:
        support_images, support_fg_mask, case = get_support_set_polyps(_config, tr_dataset)
        
    with tqdm(testloader) as pbar: 
        for idx, sample_batched in enumerate(tqdm(testloader)):
            case = sample_batched['case'][0]
            if is_alp_ds and not uses_dino_support_pool(support_selection):
                if _config.get("ablation_mode") == "presence_score":
                    if len(all_support_images) != 1:
                        raise RuntimeError(
                            "presence_score requires exactly one support part; "
                            "run with task.npart=1"
                        )
                    if qpart != 0:
                        support_images, support_fg_mask = update_support_set_by_scan_part(
                            all_support_images, all_support_fg_mask, 0
                        )
                        qpart = 0
                else:
                    support_images, support_fg_mask, qpart = manage_support_sets(
                        sample_batched,
                        all_support_images,
                        all_support_fg_mask,
                        support_images,
                        support_fg_mask,
                        qpart,
                    )
            
            if is_alp_ds and sample_batched["scan_id"][0] in support_scan_id:
                continue
             
            query_images = sample_batched['image'].cuda()
            query_labels = torch.cat([sample_batched['label']], dim=0)
            gt_present = bool((query_labels > 0).any().item())
            if not gt_present and _config["skip_no_organ_slices"]:
                continue

            qspa_weights = None
            qspa_info = None
            if is_polyp_ds and polyp_pool_paths is not None and polyp_pool_by_dataset is not None:
                query_dataset = resolve_query_dataset(sample_batched)
                pool_indices = None
                if polyp_match_support:
                    pool_indices, _ = resolve_pool_indices(
                        polyp_pool_by_dataset, query_dataset, polyp_unmatched_policy)
                    if pool_indices is None:
                        polyp_support_skipped += 1
                        continue

                if uses_dino_support_pool(support_selection):
                    support_images, support_fg_mask, _, qspa_weights, qspa_info = select_supports_by_dino_sim(
                        fewshot_seg, query_images, polyp_support_features,
                        support_selection=support_selection,
                        top_k=qspa_top_k, temperature=qspa_temp,
                        pool_paths=polyp_pool_paths, dataset=tr_dataset,
                        pool_indices=pool_indices, query_dataset=query_dataset,
                        retrieval_mode=support_retrieval_mode,
                        spatial_chunk_size=spatial_chunk_size)
                    qspa_weights = coarse_support_weights(
                        support_selection, qspa_weights, qspa_info["top_k"])
                    model.last_qspa_info = {
                        k: v for k, v in qspa_info.items() if k != "all_similarities"
                    }
                    if _config.get("debug"):
                        _log.info(
                            f"QSPA ranked top-{qspa_info['top_k']} dataset={query_dataset}: {qspa_info['ranked']}"
                        )
                elif polyp_match_support:
                    support_images, support_fg_mask, match_info = select_random_matched_support(
                        polyp_pool_paths,
                        polyp_pool_by_dataset,
                        query_dataset,
                        tr_dataset.load_support_item,
                        n_support=_config.get("n_support", 1),
                        seed=_config.get("seed", 42),
                        query_index=idx,
                        unmatched_policy=polyp_unmatched_policy,
                    )
                    if support_images is None:
                        polyp_support_skipped += 1
                        continue
                    if _config.get("debug"):
                        _log.info(
                            f"Matched random support dataset={query_dataset} "
                            f"indices={match_info['selected_indices']}"
                        )
            elif is_alp_ds and uses_dino_support_pool(support_selection) and alp_load_support_fn is not None:
                support_images, support_fg_mask, _, qspa_weights, qspa_info = select_supports_by_dino_sim(
                    fewshot_seg, query_images, polyp_support_features,
                    support_selection=support_selection,
                    top_k=qspa_top_k, temperature=qspa_temp,
                    retrieval_mode=support_retrieval_mode,
                    spatial_chunk_size=spatial_chunk_size,
                    load_support_fn=alp_load_support_fn,
                    query_dataset=str(case),
                )
                qspa_weights = coarse_support_weights(
                    support_selection, qspa_weights, qspa_info["top_k"])
                model.last_qspa_info = {
                    k: v for k, v in qspa_info.items() if k != "all_similarities"
                }
            
            n_try = 1
            with torch.no_grad():
                # MULTI_SUPPORT_PATCHED
                if _config.get("ablation_mode") == "multi_support" and is_alp_ds:
                    coarse_model_inputs = []
                    for _p in range(len(all_support_images)):
                        _si, _sl = update_support_set_by_scan_part(all_support_images, all_support_fg_mask, _p)
                        _cmi = InputFactory.create_input(
                                input_type=_config["base_model"],
                                query_image=query_images,
                                support_images=_si,
                                support_labels=_sl,
                                isval=True,
                                val_wsize=_config["val_wsize"],
                                original_sz=query_images.shape[-2:],
                                img_sz=query_images.shape[-2:],
                                gts=query_labels,
                        )
                        _cmi.to(torch.device("cuda"))
                        coarse_model_inputs.append(_cmi)
                    query_pred, scores = model(
                        query_images, coarse_model_inputs, degrees_rotate=0,
                        gt_mask=query_labels[0].to(query_images.device))
                else:
                    coarse_model_input = InputFactory.create_input(
                                            input_type=_config["base_model"],
                                            query_image=query_images,
                                            support_images=support_images,
                                            support_labels=support_fg_mask,
                                            isval=True,
                                            val_wsize=_config["val_wsize"],
                                            original_sz=query_images.shape[-2:],
                                            img_sz=query_images.shape[-2:],
                                            gts=query_labels,
                                            support_weights=qspa_weights,
                    )
                    coarse_model_input.to(torch.device("cuda"))
                        
                    query_pred, scores = model(
                            query_images, coarse_model_input, degrees_rotate=0,
                            gt_mask=query_labels[0].to(query_images.device)
                            if (
                                _config.get("ablation_mode", "none") != "none"
                                or _config.get("candidate_audit", False)
                            ) else None)
            query_pred = query_pred.cpu().detach()
            pred_present = bool((query_pred > 0).any().item())
            if _config.get("record_prediction_manifest", False):
                binary_prediction = np.ascontiguousarray(
                    (query_pred > 0).numpy().astype(np.uint8)
                )
                prediction_rows.append({
                    "dataset": _config["dataset"],
                    "organ": _config["curr_cls"],
                    "matcher": _config.get("clsname", ""),
                    "pipeline": _config.get("ablation_mode", "none"),
                    "fold": _config.get("eval_fold", ""),
                    "case": case,
                    "scan_id": _plain_sample_value(sample_batched, "scan_id"),
                    "z_id": _plain_sample_value(sample_batched, "z_id", idx),
                    "sample_index": idx,
                    "gt_present": gt_present,
                    "predicted_area_pixels": int(binary_prediction.sum()),
                    "prediction_sha256": hashlib.sha256(
                        binary_prediction.tobytes()
                    ).hexdigest(),
                })
            presence_scores = getattr(model, "last_presence_scores", None)
            if presence_scores is not None:
                final_area = int((query_pred > 0).sum().item())
                model_config = _config.get("model", {})
                presence_rows.append({
                    "dataset": _config["dataset"],
                    "organ": _config["curr_cls"],
                    "matcher": _config.get(
                        "clsname",
                        model_config.get("cls_name", "")
                        if isinstance(model_config, dict) else "",
                    ),
                    "pipeline": _config.get("ablation_mode", "none"),
                    "support_strategy": "middle_positive_single",
                    "support_npart": _config.get("task", {}).get("npart", ""),
                    "fold": _config.get("eval_fold", ""),
                    "case": case,
                    "scan_id": _plain_sample_value(sample_batched, "scan_id"),
                    "z_id": _plain_sample_value(sample_batched, "z_id", idx),
                    "sample_index": idx,
                    "gt_present": gt_present,
                    "pred_present": pred_present,
                    "final_predicted_area": final_area,
                    "final_predicted_area_ratio": final_area / float(query_pred.numel()),
                    **presence_scores,
                })

            audit_metadata = {
                "dataset": _config["dataset"],
                "organ": _config["curr_cls"],
                "matcher": _config.get("clsname", ""),
                "pipeline": _config.get("ablation_mode", "none"),
                "fold": _config.get("eval_fold", ""),
                "case": case,
                "scan_id": _plain_sample_value(sample_batched, "scan_id"),
                "z_id": _plain_sample_value(sample_batched, "z_id", idx),
                "sample_index": idx,
            }
            for row in getattr(model, "last_candidate_proposals", None) or []:
                candidate_rows.append({**audit_metadata, **row})
            for row in getattr(model, "last_proposal_slices", None) or []:
                proposal_slice_rows.append({**audit_metadata, **row})
                
            if _config["debug"]:
                if is_alp_ds:
                    save_path = f'debug/preds/{case}_{sample_batched["z_id"].item()}_{idx}_{n_try}.png'
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                elif is_polyp_ds:
                    save_path = f'debug/preds/{case}_{idx}_{n_try}.png'
                plot_pred_gt_support(query_images[0,0].cpu(), query_pred.cpu(), query_labels[0].cpu(
                ), support_images, support_fg_mask, save_path=save_path, score=scores[0])

            metrics = None
            if gt_present:
                metrics = get_dice_iou_precision_recall(
                    query_pred, query_labels[0].to(query_pred.device))
                boundary_metrics = get_boundary_metrics(
                    query_pred, query_labels[0], tolerance_px=2.0)
                metrics.update(boundary_metrics)
                mean_dice.append(metrics["dice"])
                mean_prec.append(metrics["precision"])
                mean_rec.append(metrics["recall"])
                mean_iou.append(metrics["iou"])
                mean_boundary_dice.append(metrics["boundary_dice"])
                mean_boundary_rec.append(metrics["boundary_recall"])
                mean_hd95.append(metrics["hd95"])

                bboxes_w_scores.append({"pred_bbox": get_bounding_box(query_pred.cpu()),
                                        "gt_bbox": get_bounding_box(query_labels[0].cpu()),
                                        "score": np.mean(scores)})

                if case not in mean_dice_cases:
                    mean_dice_cases[case] = []
                    mean_iou_cases[case] = []
                mean_dice_cases[case].append(metrics["dice"])
                mean_iou_cases[case].append(metrics["iou"])

            if metrics is not None and metrics["dice"] < 0.6 and _config["debug"]:
                path = f'{_run.observers[0].dir}/bad_preds/case_{case}_idx_{idx}_dice_{metrics["dice"]:.4f}.png'
                if _config["debug"]:
                    path = f'debug/bad_preds/case_{case}_idx_{idx}_dice_{metrics["dice"]:.4f}.png'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                print(f"saving bad prediction to {path}")
                plot_pred_gt_support(query_images[0,0].cpu(), query_pred.cpu(), query_labels[0].cpu(
                    ), support_images, support_fg_mask, save_path=path, score=scores[0])
                
            postfix = {
                "positive_mdice": f"{_safe_mean(mean_dice):.4f}",
                "positive_miou": f"{_safe_mean(mean_iou):.4f}, n_try: {n_try}",
            }
            if uses_dino_support_pool(support_selection) and qspa_info is not None:
                ranked0 = qspa_info["ranked"][0]
                postfix["supp"] = f"{ranked0[0]}:{ranked0[1]:.3f}x{len(qspa_info['ranked'])}"
            pbar.set_postfix_str(postfix)
                

    if is_polyp_ds and polyp_match_support and polyp_support_skipped:
        _log.info(
            f'Polyp matched-support skipped {polyp_support_skipped} queries '
            f'(no train pool; policy={polyp_unmatched_policy})'
        )

    for k in mean_dice_cases.keys():
        _run.log_scalar(f'mar_val_batches_meanDice_{k}', np.mean(mean_dice_cases[k]))
        _run.log_scalar(f'mar_val_batches_meanIOU_{k}', np.mean(mean_iou_cases[k]))
        _log.info(f'mar_val batches meanDice_{k}: {np.mean(mean_dice_cases[k])}')
        _log.info(f'mar_val batches meanIOU_{k}: {np.mean(mean_iou_cases[k])}') 
    
    # write validation result to log file
    m_meanDice = _safe_mean(mean_dice)
    m_meanPrec = _safe_mean(mean_prec)
    m_meanRec = _safe_mean(mean_rec)
    m_meanIOU = _safe_mean(mean_iou)
    m_meanBoundaryDice = _safe_mean(mean_boundary_dice)
    m_meanBoundaryRec = _safe_mean(mean_boundary_rec)
    m_meanHD95 = _safe_mean(mean_hd95)

    _run.log_scalar('mar_val_batches_meanDice', m_meanDice)
    _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec)
    _run.log_scalar('mar_val_al_batches_meanRec', m_meanRec)
    _run.log_scalar('mar_val_al_batches_meanIOU', m_meanIOU)
    _run.log_scalar('mar_val_batches_meanBoundaryDice', m_meanBoundaryDice)
    _run.log_scalar('mar_val_batches_meanBoundaryRecall', m_meanBoundaryRec)
    _run.log_scalar('mar_val_batches_meanHD95', m_meanHD95)
    _log.info(f'mar_val batches meanDice: {m_meanDice}')
    _log.info(f'mar_val batches meanPrec: {m_meanPrec}')
    _log.info(f'mar_val batches meanRec: {m_meanRec}')
    _log.info(f'mar_val batches meanIOU: {m_meanIOU}')
    _log.info(f'mar_val batches meanBoundaryDice@2px: {m_meanBoundaryDice}')
    _log.info(f'mar_val batches meanBoundaryRecall@2px: {m_meanBoundaryRec}')
    _log.info(f'mar_val batches meanHD95(px): {m_meanHD95}')

    if presence_rows:
        annotate_presence_slice_distance(presence_rows)
        output_dir = _run.observers[0].dir if _run.observers else "."
        presence_csv = os.path.join(output_dir, "presence_scores.csv")
        with open(presence_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(presence_rows[0].keys()))
            writer.writeheader()
            writer.writerows(presence_rows)
        _log.info(f"presence per-slice scores: {presence_csv}")

        mask_metrics = mask_presence_metrics(presence_rows)
        ranking_all = ranking_metrics(presence_rows, PRESENCE_SCORE_NAMES)
        ranking_hard3 = ranking_metrics(
            presence_rows,
            PRESENCE_SCORE_NAMES,
            negative_filter=lambda row: bool(row.get("hard_negative_3", False)),
        )
        for name, value in mask_metrics.items():
            if is_finite_number(value):
                _run.log_scalar(f"presence_mask_{name}", float(value))
                _log.info(f"presence mask {name}: {value}")
        for score_name in PRESENCE_SCORE_NAMES:
            for metric_name in ("auroc", "auprc"):
                value = ranking_all[score_name][metric_name]
                if is_finite_number(value):
                    _run.log_scalar(
                        f"presence_{score_name}_{metric_name}", float(value)
                    )
                    _log.info(f"presence {score_name} {metric_name}: {value}")
                hard_value = ranking_hard3[score_name][metric_name]
                if is_finite_number(hard_value):
                    _run.log_scalar(
                        f"presence_hard3_{score_name}_{metric_name}",
                        float(hard_value),
                    )
                    _log.info(
                        f"presence hard-negative@3 {score_name} {metric_name}: "
                        f"{hard_value}"
                    )

    if proposal_slice_rows:
        output_dir = _run.observers[0].dir if _run.observers else "."
        candidate_csv = os.path.join(output_dir, "candidate_proposals.csv")
        if candidate_rows:
            with open(candidate_csv, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(candidate_rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(candidate_rows)
        slice_csv = os.path.join(output_dir, "proposal_slices.csv")
        with open(slice_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(proposal_slice_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(proposal_slice_rows)
        proposal_summary = summarize_proposal_slices(proposal_slice_rows)
        summary_csv = os.path.join(output_dir, "candidate_audit_summary.csv")
        with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(proposal_summary[0].keys()))
            writer.writeheader()
            writer.writerows(proposal_summary)

    if prediction_rows:
        output_dir = _run.observers[0].dir if _run.observers else "."
        manifest_csv = os.path.join(output_dir, "prediction_manifest.csv")
        with open(manifest_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(prediction_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(prediction_rows)
        _log.info(f"prediction manifest: {manifest_csv}")
    print("============ ============")
    _log.info(f'End of validation')
    return 1

# # ABLATION_V2_PATCHED
