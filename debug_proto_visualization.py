import argparse
import os
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from models.grid_proto_fewshot import FewShotSeg
from util import proto_debug_viz as debug_viz


def read_image(path: str, image_size: int) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def read_mask(path: str, image_size: int) -> torch.Tensor:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)


def build_episode_from_images(args) -> Tuple[list, list, list, list]:
    supp_img = read_image(args.support_image, args.image_size)
    qry_img = read_image(args.query_image, args.image_size)
    fg = read_mask(args.support_mask, args.image_size)
    bg = 1.0 - fg
    return [[supp_img]], [[fg]], [[bg]], [qry_img]


def load_episode(path: str) -> Tuple[list, list, list, list]:
    data = torch.load(path, map_location='cpu')
    if isinstance(data, (list, tuple)) and len(data) >= 4:
        return data[0], data[1], data[2], data[3]
    required = ['supp_imgs', 'fore_mask', 'back_mask', 'qry_imgs']
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f'Episode file missing keys: {missing}. Required keys: {required}')
    return data['supp_imgs'], data['fore_mask'], data['back_mask'], data['qry_imgs']


def move_episode_to_device(supp_imgs, fore_mask, back_mask, qry_imgs, device):
    supp_imgs = [[x.to(device) for x in way] for way in supp_imgs]
    fore_mask = [[x.to(device) for x in way] for way in fore_mask]
    back_mask = [[x.to(device) for x in way] for way in back_mask]
    qry_imgs = [x.to(device) for x in qry_imgs]
    return supp_imgs, fore_mask, back_mask, qry_imgs


def default_model_cfg(args) -> Dict:
    return {
        'align': False,
        'debug': False,
        'debug_viz': True,
        'debug_dir': args.out_dir,
        'use_coco_init': args.use_coco_init,
        'which_model': args.backbone,
        'cls_name': args.cls_name,
        'proto_grid_size': args.proto_grid_size,
        'feature_hw': [args.image_size // 8, args.image_size // 8],
        'reload_model_path': args.checkpoint,
        'lora': 0,
        'use_slice_adapter': False,
        'adapter_layers': 3,
        'use_pos_enc': False,
        'gate_tau': args.gate_tau,
        'bootstrap_thresh': args.bootstrap_thresh,
        'bootstrap_mode': args.bootstrap_mode,
        'topk_ratio': args.topk_ratio,
        'spen_kmax': args.spen_kmax,
        'spen_cs': args.spen_cs,
        'sinkhorn_iters': args.sinkhorn_iters,
        'sinkhorn_eps': args.sinkhorn_eps,
    }


def save_prediction(output: torch.Tensor, qry_img: torch.Tensor, out_dir: str) -> None:
    prob = torch.softmax(output, dim=1)
    pred = prob[:, 1].detach().float()
    debug_viz.save_heatmap(pred[0], os.path.join(out_dir, 'prediction_fg_probability_heatmap.png'), image=qry_img[0])
    mask = output.argmax(dim=1).detach().float()[0]
    debug_viz.save_mask_overlay(qry_img[0], mask, os.path.join(out_dir, 'prediction_mask_overlay.png'), color_bgr=(0, 0, 255), alpha=0.45)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate ALPNet/SPEN prototype debug visualizations.')
    parser.add_argument('--episode', default=None, help='Optional .pt/.pth file with supp_imgs/fore_mask/back_mask/qry_imgs.')
    parser.add_argument('--support-image', default=None, help='Support RGB image path.')
    parser.add_argument('--support-mask', default=None, help='Support foreground mask path.')
    parser.add_argument('--query-image', default=None, help='Query RGB image path.')
    parser.add_argument('--out-dir', default='debug/proto_debug', help='Directory to save debug images.')
    parser.add_argument('--checkpoint', default=None, help='Optional FewShotSeg checkpoint path.')
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--backbone', default='dinov2_b14', choices=['dlfcn_res101', 'default', 'dinov2_b14', 'dinov2_l14', 'dinov2_l14_reg'])
    parser.add_argument('--cls-name', default='grid_proto', help='Use grid_proto for ALPNet, or spen_full/spen_alpg/spen_qlpe/spen_qgpg for SPEN.')
    parser.add_argument('--proto-grid-size', type=int, default=8)
    parser.add_argument('--val-wsize', type=int, default=2)
    parser.add_argument('--use-coco-init', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gate-tau', type=float, default=2.0)
    parser.add_argument('--bootstrap-thresh', type=float, default=0.5)
    parser.add_argument('--bootstrap-mode', default='threshold', choices=['threshold', 'topk'])
    parser.add_argument('--topk-ratio', type=float, default=0.2)
    parser.add_argument('--spen-kmax', type=int, default=24, help='SPENet paper default: 24.')
    parser.add_argument('--spen-cs', type=int, default=50, help='Foreground pixels per local region; paper default: 50.')
    parser.add_argument('--sinkhorn-iters', type=int, default=50)
    parser.add_argument('--sinkhorn-eps', type=float, default=0.1, help='SPENet paper default: 0.1.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.episode:
        supp_imgs, fore_mask, back_mask, qry_imgs = load_episode(args.episode)
    else:
        if not (args.support_image and args.support_mask and args.query_image):
            raise ValueError('Provide either --episode or --support-image --support-mask --query-image.')
        supp_imgs, fore_mask, back_mask, qry_imgs = build_episode_from_images(args)

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    cfg = default_model_cfg(args)
    model = FewShotSeg(args.image_size, pretrained_path=args.checkpoint, cfg=cfg).to(device)
    model.eval()
    supp_imgs, fore_mask, back_mask, qry_imgs = move_episode_to_device(supp_imgs, fore_mask, back_mask, qry_imgs, device)

    with torch.no_grad():
        output, *_ = model(supp_imgs, fore_mask, back_mask, qry_imgs, isval=True, val_wsize=args.val_wsize, show_viz=False)
    save_prediction(output.cpu(), qry_imgs[0].cpu(), args.out_dir)
    print(f'Debug visualizations saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
