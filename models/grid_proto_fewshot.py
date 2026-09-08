"""
ALPNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .alpmodule import MultiProtoAsConv
from .spen import SPENProtoMatcher  # # SPEN_PATCHED
from .dense_matcher import DenseForegroundMatcher

from .backbone.torchvision_backbones import TVDeeplabRes101Encoder
from util.consts import DEFAULT_FEATURE_SIZE
from util.lora import inject_trainable_lora
# from util.utils import load_config_from_url, plot_dinov2_fts
import math
import os
from util import proto_debug_viz as debug_viz

# Specify a local path to the repository (or use installed package instead)
FG_PROT_MODE = 'gridconv+' # using both local and global prototype
# FG_PROT_MODE = 'mask'
# using local prototype only. Also 'mask' refers to using global prototype only (as done in vanilla PANet)
BG_PROT_MODE = 'gridconv'

# thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95


class FewShotSeg(nn.Module):
    """
    ALPNet
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """
    # 初始化小样本分割模型，处理图像尺寸 image_size 及配置文件 cfg，并实例化特征提取器和分类器模块。
    def __init__(self, image_size, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.image_size = image_size
        self.pretrained_path = pretrained_path
        self.config = cfg or {
            'align': False, 'debug': False}
        self.get_encoder()
        self.get_cls()
        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path), strict=True)
            print(
                f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

    #该函数根据配置文件初始化图像特征提取主干网络（如 ResNet101 或 DINOv2 ），并处理特征图尺寸 feature_hw 及可选的 LoRA 参数注入
    def get_encoder(self):
        self.config['feature_hw'] = [DEFAULT_FEATURE_SIZE,
                                     DEFAULT_FEATURE_SIZE]  # default feature map size
        if self.config['which_model'] == 'dlfcn_res101' or self.config['which_model'] == 'default':
            use_coco_init = self.config['use_coco_init']
            self.encoder = TVDeeplabRes101Encoder(use_coco_init)
            self.config['feature_hw'] = [
                math.ceil(self.image_size/8), math.ceil(self.image_size/8)]
        elif self.config['which_model'] == 'dinov2_l14':
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitl14')
            self.config['feature_hw'] = [max(
                self.image_size//14, DEFAULT_FEATURE_SIZE), max(self.image_size//14, DEFAULT_FEATURE_SIZE)]
        elif self.config['which_model'] == 'dinov2_l14_reg':
            try:
                self.encoder = torch.hub.load(
                    'facebookresearch/dinov2', 'dinov2_vitl14_reg')
            except RuntimeError as e:
                self.encoder = torch.hub.load(
                    'facebookresearch/dino', 'dinov2_vitl14_reg', force_reload=True)
            self.config['feature_hw'] = [max(
                self.image_size//14, DEFAULT_FEATURE_SIZE), max(self.image_size//14, DEFAULT_FEATURE_SIZE)]
        elif self.config['which_model'] == 'dinov2_b14':
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitb14')
            self.config['feature_hw'] = [max(
                self.image_size//14, DEFAULT_FEATURE_SIZE), max(self.image_size//14, DEFAULT_FEATURE_SIZE)]
        else:
            raise NotImplementedError(
                f'Backbone network {self.config["which_model"]} not implemented')

        if self.config['lora'] > 0:
            self.encoder.requires_grad_(False)
            print(f'Injecting LoRA with rank:{self.config["lora"]}')
            encoder_lora_params = inject_trainable_lora(
                self.encoder, r=self.config['lora'])

    #该函数接收拼接后的图像张量 imgs_concat，通过编码器进行前向传播以提取空间特征图，并统一调整特征的维度和尺寸返回 img_fts
    def get_features(self, imgs_concat):
        if self.config['which_model'] == 'dlfcn_res101':
            img_fts = self.encoder(imgs_concat, low_level=False)
        elif 'dino' in self.config['which_model']:
            # resize imgs_concat to the closest size that is divisble by 14
            imgs_concat = F.interpolate(imgs_concat, size=(
                self.image_size // 14 * 14, self.image_size // 14 * 14), mode='bilinear')
            dino_fts = self.encoder.forward_features(imgs_concat)
            img_fts = dino_fts["x_norm_patchtokens"]  # B, HW, C
            img_fts = img_fts.permute(0, 2, 1)  # B, C, HW
            C, HW = img_fts.shape[-2:]
            img_fts = img_fts.view(-1, C, int(HW**0.5),
                                   int(HW**0.5))  # B, C, H, W
            if HW < DEFAULT_FEATURE_SIZE ** 2:
                img_fts = F.interpolate(img_fts, size=(
                    DEFAULT_FEATURE_SIZE, DEFAULT_FEATURE_SIZE), mode='bilinear')  # this is if h,w < (32,32)
        else:
            raise NotImplementedError(
                f'Backbone network {self.config["which_model"]} not implemented')
        
        return img_fts

    # 该函数基于预设的网格大小和特征维度 embed_dim，实例化用于计算相似度的原型分类器单元 cls_unit
    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"]

        if self.config['cls_name'] == 'grid_proto':
            embed_dim = 256
            if 'dinov2_b14' in self.config['which_model']:
                embed_dim = 768
            elif 'dinov2_l14' in self.config['which_model']:
                embed_dim = 1024
            self.cls_unit = MultiProtoAsConv(proto_grid=[proto_hw, proto_hw], feature_hw=self.config["feature_hw"], embed_dim=embed_dim)  # when treating it as ordinary prototype
            print(f"cls unit feature hw: {self.cls_unit.feature_hw}")
        elif self.config['cls_name'].startswith('dense_fg_'):
            embed_dim = 768 if 'dinov2_b14' in self.config['which_model'] else 1024
            self.cls_unit = DenseForegroundMatcher(
                proto_grid=[proto_hw, proto_hw],
                feature_hw=self.config["feature_hw"],
                embed_dim=embed_dim,
                dense_mode=self.config['cls_name'],
                temperature=self.config.get('dense_temperature', 0.05),
                hard_threshold=self.config.get('dense_hard_threshold', FG_THRESH),
                score_scale=self.config.get('dense_score_scale', 20.0),
                query_chunk_size=self.config.get('dense_query_chunk_size', 512),
                global_anchor=self.config.get('dense_global_anchor', True),
                null_strength=self.config.get('dense_null_strength', 1.0),
            )
            print(
                f"Dense foreground cls unit: mode={self.config['cls_name']}, "
                f"tau={self.cls_unit.temperature}, hard_threshold={self.cls_unit.hard_threshold}, "
                f"aggregation={self.cls_unit.aggregation}, "
                f"global_anchor={self.cls_unit.global_anchor}"
            )
        elif self.config['cls_name'].startswith('spen'):
            embed_dim = 1024
            if 'dinov2_b14' in self.config['which_model']:
                embed_dim = 768
            elif 'dinov2_l14' in self.config['which_model']:
                embed_dim = 1024
            _spen_mode = self.config['cls_name']
            _gate_tau = self.config.get('gate_tau', 2.0)
            _bootstrap_thresh = self.config.get('bootstrap_thresh', 0.5)
            _bootstrap_mode = self.config.get('bootstrap_mode', 'threshold')
            _topk_ratio = self.config.get('topk_ratio', 0.2)
            _spen_kmax = self.config.get('spen_kmax', 24)
            _spen_cs = self.config.get('spen_cs', 50)
            _sinkhorn_iters = self.config.get('sinkhorn_iters', 50)
            _sinkhorn_eps = self.config.get('sinkhorn_eps', 0.1)
            _ot_prior_strength = self.config.get('ot_prior_strength', 1.0)
            _spen_reference_size = self.config.get('spen_reference_size', 256)
            _spen_scale_cs = self.config.get('spen_scale_cs', True)
            _spen_cap_by_feature = self.config.get('spen_cap_by_feature', True)
            _spen_deterministic_fps = self.config.get('spen_deterministic_fps', True)
            self.cls_unit = SPENProtoMatcher(
                proto_grid=[proto_hw, proto_hw],
                feature_hw=self.config["feature_hw"],
                embed_dim=embed_dim,
                spen_mode=_spen_mode,
                k_max=_spen_kmax,
                Cs=_spen_cs,
                gate_tau=_gate_tau,
                bootstrap_thresh=_bootstrap_thresh,
                bootstrap_mode=_bootstrap_mode,
                topk_ratio=_topk_ratio,
                sinkhorn_iters=_sinkhorn_iters,
                sinkhorn_eps=_sinkhorn_eps,
                ot_prior_strength=_ot_prior_strength,
                reference_mask_size=_spen_reference_size,
                scale_cs=_spen_scale_cs,
                cap_by_feature_cells=_spen_cap_by_feature,
                deterministic_fps=_spen_deterministic_fps,
            )
            print(f'SPEN cls unit: mode={_spen_mode}, k_max={_spen_kmax}, Cs={_spen_cs}, tau={_gate_tau}, thresh={_bootstrap_thresh}, bootstrap_mode={_bootstrap_mode}, topk_ratio={_topk_ratio}')
        else:
            raise NotImplementedError(
                f'Classifier {self.config["cls_name"]} not implemented')

    #该函数处理多分辨率列表 resolutions，将支持集图像、查询集图像及掩码插值到不同分辨率后分别调用 forward 进行预测。
    def forward_resolutions(self, resolutions, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz=False, supp_fts=None):
        predictions = []
        for res in resolutions:
            supp_imgs_resized = [[F.interpolate(supp_img[0], size=(
                res, res), mode='bilinear') for supp_img in supp_imgs]] if supp_imgs[0][0].shape[-1] != res else supp_imgs
            fore_mask_resized = [[F.interpolate(fore_mask_way[0].unsqueeze(0), size=(res, res), mode='bilinear')[
                0] for fore_mask_way in fore_mask]] if fore_mask[0][0].shape[-1] != res else fore_mask
            back_mask_resized = [[F.interpolate(back_mask_way[0].unsqueeze(0), size=(res, res), mode='bilinear')[
                0] for back_mask_way in back_mask]] if back_mask[0][0].shape[-1] != res else back_mask
            qry_imgs_resized = [F.interpolate(qry_img, size=(res, res), mode='bilinear')
                                for qry_img in qry_imgs] if qry_imgs[0][0].shape[-1] != res else qry_imgs

            pred = self.forward(supp_imgs_resized, fore_mask_resized, back_mask_resized,
                                qry_imgs_resized, isval, val_wsize, show_viz, supp_fts)[0]
            predictions.append(pred)

    #该函数利用双线性插值，将输入的各组图像和掩码张量统一缩放至模型配置的标准图像尺寸 image_size。  
    def resize_inputs_to_image_size(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        supp_imgs = [[F.interpolate(supp_img, size=(
            self.image_size, self.image_size), mode='bilinear') for supp_img in supp_imgs_way] for supp_imgs_way in supp_imgs]
        fore_mask = [[F.interpolate(fore_mask_way[0].unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear')[
            0] for fore_mask_way in fore_mask]] if fore_mask[0][0].shape[-1] != self.image_size else fore_mask
        back_mask = [[F.interpolate(back_mask_way[0].unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear')[
            0] for back_mask_way in back_mask]] if back_mask[0][0].shape[-1] != self.image_size else back_mask
        qry_imgs = [F.interpolate(qry_img, size=(self.image_size, self.image_size), mode='bilinear')
                    for qry_img in qry_imgs] if qry_imgs[0][0].shape[-1] != self.image_size else qry_imgs
        return supp_imgs, fore_mask, back_mask, qry_imgs

    #该函数执行模型的核心前向传播，处理支持集和查询集数据以生成原型并计算相似度，最终输出分割预测结果张量及对齐损失
    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz=False, supp_fts=None):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            show_viz: return the visualization dictionary
        """
        # ('Please go through this piece of code carefully')
        # supp_imgs, fore_mask, back_mask, qry_imgs = self.resize_inputs_to_image_size(
        #     supp_imgs, fore_mask, back_mask, qry_imgs)
        '''
        步骤 1：维度验证与变量初始化。 函数首先获取支持集的类别数（n_ways）、样本数（n_shots）以及查询集数量，
        并断言当前仅支持单样本（1-way）和单查询（1-query）模式，同时提取基础的批次大小和图像维度信息。
        '''
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        # NOTE: actual shot in support goes in batch dimension
        assert n_ways == 1, "Multi-shot has not been implemented yet"
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        if self.config["cls_name"] == 'grid_proto_3d':
            img_size = supp_imgs[0][0].shape[-3:]
        qry_bsize = qry_imgs[0].shape[0]

        '''
        步骤 2：图像拼接与特征提取。 将支持集图像和查询集图像在批次维度拼接成 imgs_concat，
        调用 get_features 统一提取它们的深度特征图 img_fts
        '''
        #统一提取特征。 模型首先将支持集图像和查询集图像拼接在一起，送入骨干网络（如 ResNet 或 DINOv2）
        # 提取深度特征图。
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)

        img_fts = self.get_features(imgs_concat)

        #步骤 3：特征分离与重塑。 根据索引将提取的全局特征图切分，分别重塑为支持集特征张量 supp_fts（包含 way、shot、batch 维度）和查询集特征张量 qry_fts。
        if len(img_fts.shape) == 5:  # for 3D
            fts_size = img_fts.shape[-3:]
        else:
            fts_size = img_fts.shape[-2:]
        if supp_fts is None:
            supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
                n_ways, n_shots, sup_bsize, -1, *fts_size)  # wa x sh x b x c x h' x w'
            qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
                n_queries, qry_bsize, -1, *fts_size)   # N x B x C x H' x W'
        else:
            # N x B x C x H' x W'
            qry_fts = img_fts.view(n_queries, qry_bsize, -1, *fts_size)

        debug_enabled = bool(self.config.get('debug_viz', False) or show_viz)
        debug_dir = self.config.get('debug_dir', os.path.join('debug', 'proto_debug'))
        if debug_enabled:
            os.makedirs(debug_dir, exist_ok=True)
            debug_viz.save_feature_pca(supp_fts[0, 0, 0], os.path.join(debug_dir, 'alpnet_02_supp_fts_pca.png'), target_size=img_size)
            debug_viz.save_feature_pca(qry_fts[0, 0], os.path.join(debug_dir, 'alpnet_02_qry_fts_pca.png'), target_size=img_size)

        '''
        步骤 4：掩码处理。 将支持集的前景掩码 fore_mask 和背景掩码 back_mask 堆叠为高维张量，
        并设置前景掩码张量需要追踪梯度（requires_grad=True）。
        '''
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad=True)
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        visualizes = []  # the buffer for visualization

        for epi in range(1):  # batch dimension, fixed to 1
            fg_masks = []  # keep the way part

            '''
            for way in range(n_ways):
                # note: index of n_ways starts from 0
                mean_sup_ft = supp_fts[way].mean(dim = 0) # [ nb, C, H, W]. Just assume batch size is 1 as pytorch only allows this
                mean_sup_msk = F.interpolate(fore_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear')
                fg_masks.append( mean_sup_msk )

                mean_bg_msk = F.interpolate(back_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear') # [nb, C, H, W]
            '''
            '''
            步骤 5：掩码降采样。 使用最近邻插值（nearest），将前景和背景掩码的尺寸缩放至与提取的特征图尺寸（fts_size）完全一致，
            得到 res_fg_msk 和 res_bg_msk，以便后续计算原型。
            '''
            # re-interpolate support mask to the same size as support feature
            if len(fts_size) == 3:  # TODO make more generic
                res_fg_msk = torch.stack([F.interpolate(fore_mask[0][0].unsqueeze(
                    0), size=fts_size, mode='nearest')], dim=0)  # [nway, ns, nb, nd', nh', nw'])
                res_bg_msk = torch.stack([F.interpolate(back_mask[0][0].unsqueeze(
                    0), size=fts_size, mode='nearest')], dim=0)  # [nway, ns, nb, nd', nh', nw'])
            else:
                res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size=fts_size, mode='nearest')
                                         for fore_mask_w in fore_mask], dim=0)  # [nway, ns, nb, nh', nw']
                res_bg_msk = torch.stack([F.interpolate(back_mask_w, size=fts_size, mode='nearest')
                                         for back_mask_w in back_mask], dim=0)  # [nway, ns, nb, nh', nw']

            if debug_enabled:
                debug_viz.save_mask_overlay(supp_imgs[0][0][epi], res_fg_msk[0, 0, epi], os.path.join(debug_dir, 'alpnet_01_res_fg_msk_overlay.png'), color_bgr=(0, 0, 255), alpha=0.5)
                debug_viz.save_mask_overlay(supp_imgs[0][0][epi], res_bg_msk[0, 0, epi], os.path.join(debug_dir, 'alpnet_01_res_bg_msk_overlay.png'), color_bgr=(255, 0, 0), alpha=0.5)

            scores = []
            assign_maps = []
            bg_sim_maps = []
            fg_sim_maps = []
            bg_mode = BG_PROT_MODE

            '''
            步骤 6：背景相似度计算。 将查询特征、支持特征与降采样后的背景掩码输入分类器单元 cls_unit，
            基于背景原型模式 BG_PROT_MODE 计算并保存背景像素的相似度得分 _raw_score 及分配图。
            '''
            _raw_score, _, aux_attr, _ = self.cls_unit(
                qry_fts, supp_fts, res_bg_msk, mode=bg_mode, thresh=BG_THRESH, isval=isval, val_wsize=val_wsize, vis_sim=debug_enabled)
            scores.append(_raw_score)
            assign_maps.append(aux_attr['proto_assign'])
            if debug_enabled:
                debug_viz.save_heatmap(_raw_score[0, 0], os.path.join(debug_dir, 'alpnet_04_raw_score_bg_heatmap.png'), target_size=img_size, image=qry_imgs[0][epi])
            
            for way, _msks in enumerate(res_fg_msk):
                raw_scores = []
                for i, _msk in enumerate(_msks):
                    _msk = _msk.unsqueeze(0)
                    supp_ft = supp_fts[:, i].unsqueeze(0)
                    if self.config["cls_name"] == 'grid_proto_3d':  # 3D
                        k_size = self.cls_unit.kernel_size
                        fg_mode = FG_PROT_MODE if F.avg_pool3d(_msk, k_size).max(
                        ) >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'  # TODO figure out kernel size
                    else:
                        k_size = self.cls_unit.kernel_size
                        _handles_own_fg = self.config.get('cls_name', '').startswith(('spen', 'dense_fg_'))
                        if _handles_own_fg:
                            fg_mode = 'mask'  # matcher handles its own foreground representation
                        else:
                            fg_mode = FG_PROT_MODE if F.avg_pool2d(_msk, k_size).max(
                            ) >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'
                            # TODO figure out kernel size
                    _raw_score, _, aux_attr, proto_grid = self.cls_unit(
                        qry_fts, supp_ft, _msk.unsqueeze(0), mode=fg_mode,
                        thresh=FG_THRESH, isval=isval, val_wsize=val_wsize,
                        vis_sim=debug_enabled,
                        full_res_mask=fore_mask[way, i, epi],
                        background_score=(
                            scores[0]
                            if self.config.get('cls_name', '').startswith('dense_fg_')
                            else None
                        ),
                    )
                    raw_scores.append(_raw_score)
                '''
                步骤 7：前景相似度计算与原型模式动态选择。 遍历每个前景掩码，利用平均池化激活值判断当前目标的范围大小，
                动态决定使用网格原型（gridconv+）还是全局掩码原型（mask）；随后输入分类器计算前景得分，并在多 Shot 维度取最大值得分
                '''
                # create a score where each feature is the max of the raw_score
                _raw_score = torch.stack(raw_scores, dim=1).max(dim=1)[
                    0] 
                scores.append(_raw_score)
                assign_maps.append(aux_attr['proto_assign'])
                if debug_enabled:
                    if proto_grid is not None:
                        debug_viz.save_assign_overlay(proto_grid, supp_fts[0, 0, epi], os.path.join(debug_dir, 'alpnet_03_proto_grid_overlay.png'), target_size=img_size)
                    elif aux_attr.get('proto_assign') is not None:
                        debug_viz.save_assign_overlay(aux_attr['proto_assign'], qry_fts[0, epi], os.path.join(debug_dir, 'alpnet_03_assign_maps_overlay.png'), target_size=img_size)
                    debug_viz.save_heatmap(_raw_score[0, 0], os.path.join(debug_dir, 'alpnet_04_raw_score_fg_heatmap.png'), target_size=img_size, image=qry_imgs[0][epi])
                    score_margin = _raw_score[0, 0] - scores[0][0, 0]
                    debug_viz.save_heatmap(score_margin, os.path.join(debug_dir, 'alpnet_05_fg_minus_bg_margin.png'), target_size=img_size, image=qry_imgs[0][epi])
                    if self.config.get('cls_name', '').startswith('dense_fg_'):
                        occupancy = aux_attr.get('dense_occupancy')
                        dense_similarity = aux_attr.get('dense_similarity')
                        global_similarity = aux_attr.get('global_similarity')
                        print(
                            'Dense foreground evidence:',
                            f"mode={self.config.get('cls_name')}",
                            f"effective_mass={aux_attr.get('dense_effective_token_count')}",
                            f"nonzero_tokens={aux_attr.get('dense_nonzero_token_count')}",
                            f"hard_fallback={aux_attr.get('dense_hard_fallback')}",
                            f"temperature={aux_attr.get('dense_temperature')}",
                            f"aggregation={aux_attr.get('dense_aggregation')}",
                            f"uses_null={aux_attr.get('dense_uses_background_null')}",
                            f"null_strength={aux_attr.get('dense_null_strength')}",
                        )
                        if occupancy is not None:
                            debug_viz.save_mask_overlay(
                                supp_imgs[0][0][epi], occupancy[0, 0],
                                os.path.join(debug_dir, 'dense_01_support_occupancy_overlay.png'),
                                color_bgr=(0, 165, 255), alpha=0.55,
                            )
                        if dense_similarity is not None:
                            debug_viz.save_heatmap(
                                dense_similarity[0, 0],
                                os.path.join(debug_dir, 'dense_02_lse_evidence_heatmap.png'),
                                target_size=img_size, image=qry_imgs[0][epi],
                            )
                        if global_similarity is not None:
                            debug_viz.save_heatmap(
                                global_similarity[0, 0],
                                os.path.join(debug_dir, 'dense_03_global_anchor_heatmap.png'),
                                target_size=img_size, image=qry_imgs[0][epi],
                            )
                        foreground_match_probability = aux_attr.get(
                            'dense_foreground_match_probability'
                        )
                        if foreground_match_probability is not None:
                            debug_viz.save_heatmap(
                                foreground_match_probability[0, 0],
                                os.path.join(debug_dir, 'dense_04_null_match_probability.png'),
                                target_size=img_size, image=qry_imgs[0][epi],
                            )
                        null_penalty = aux_attr.get('dense_null_penalty')
                        if null_penalty is not None:
                            debug_viz.save_heatmap(
                                null_penalty[0, 0],
                                os.path.join(debug_dir, 'dense_05_null_penalty.png'),
                                target_size=img_size, image=qry_imgs[0][epi],
                            )
                        attention_entropy = aux_attr.get('dense_attention_entropy')
                        if attention_entropy is not None:
                            debug_viz.save_heatmap(
                                attention_entropy[0, 0],
                                os.path.join(debug_dir, 'dense_06_attention_entropy.png'),
                                target_size=img_size, image=qry_imgs[0][epi],
                            )
                        dense_debug = {
                            key: value.detach().cpu() if torch.is_tensor(value) else value
                            for key, value in aux_attr.items()
                            if key.startswith('dense_') or key == 'global_similarity'
                        }
                        dense_debug.update({
                            'raw_score_bg': scores[0].detach().cpu(),
                            'raw_score_fg': _raw_score.detach().cpu(),
                            'fg_minus_bg_margin': score_margin.detach().cpu(),
                        })
                        torch.save(dense_debug, os.path.join(debug_dir, 'dense_debug_tensors.pt'))
                    if self.config.get('cls_name', '').startswith('spen'):
                        print(
                            'SPEN prototype counts:',
                            f"support_pixels={aux_attr.get('spen_n_fg')}",
                            f"support_k={aux_attr.get('spen_k')}",
                            f"support_area_k={aux_attr.get('spen_area_k')}",
                            f"support_feature_cells={aux_attr.get('spen_effective_feature_cells')}",
                            f"effective_Cs={aux_attr.get('spen_effective_cs')}",
                            f"query_pixels={aux_attr.get('qry_n_fg')}",
                            f"query_k={aux_attr.get('qry_k')}",
                            f"query_area_k={aux_attr.get('qry_area_k')}",
                            f"query_feature_cells={aux_attr.get('qry_effective_feature_cells')}",
                        )
                        if aux_attr.get('spen_fg_hw') is not None:
                            debug_viz.save_points_overlay(supp_imgs[0][0][epi], aux_attr.get('spen_fg_hw'), aux_attr.get('spen_center_idx'), aux_attr.get('spen_feature_hw', fts_size), os.path.join(debug_dir, 'spen_01_centers_overlay.png'))
                        if aux_attr.get('qry_coarse_mask') is not None:
                            debug_viz.save_mask_overlay(qry_imgs[0][epi], aux_attr['qry_coarse_mask'], os.path.join(debug_dir, 'spen_02_qry_coarse_mask_overlay.png'), color_bgr=(0, 255, 255), alpha=0.5)
                        if aux_attr.get('weights') is not None:
                            print('SPEN OT weights:', aux_attr['weights'].detach().cpu().numpy())
                            if aux_attr.get('transport_plan') is not None:
                                print('SPEN transport plan:', aux_attr['transport_plan'].detach().cpu().numpy())
                            debug_viz.save_bar(aux_attr['weights'], os.path.join(debug_dir, 'spen_03_weights_bar.png'))
                            debug_viz.save_bar(aux_attr['weights_relative'], os.path.join(debug_dir, 'spen_03_weights_relative_bar.png'))
                        if aux_attr.get('multi_proto_priors') is not None:
                            print('SPEN multi-prototype priors:', aux_attr['multi_proto_priors'].detach().cpu().numpy())
                            debug_viz.save_bar(
                                aux_attr['multi_proto_priors'],
                                os.path.join(debug_dir, 'spen_03_multi_proto_priors_bar.png')
                            )
                        if aux_attr.get('sim') is not None:
                            debug_viz.save_heatmap(aux_attr['sim'][0, 0], os.path.join(debug_dir, 'spen_04_final_sim_heatmap.png'), target_size=img_size, image=qry_imgs[0][epi])
                        tensor_debug = {
                            key: value.detach().cpu() if torch.is_tensor(value) else value
                            for key, value in aux_attr.items()
                            if key in {
                                'weights', 'weights_relative', 'prototype_similarity', 'transport_plan',
                                'transport_row_sum', 'transport_col_sum',
                                'multi_proto_priors',
                                'spen_n_fg', 'spen_k', 'qry_n_fg', 'qry_k',
                                'spen_area_k', 'spen_effective_cs',
                                'spen_effective_feature_cells', 'qry_area_k',
                                'qry_effective_cs', 'qry_effective_feature_cells',
                                'qry_bootstrap_sim', 'qry_coarse_mask'
                            }
                        }
                        tensor_debug.update({
                            'raw_score_bg': scores[0].detach().cpu(),
                            'raw_score_fg': _raw_score.detach().cpu(),
                            'fg_minus_bg_margin': score_margin.detach().cpu(),
                        })
                        torch.save(tensor_debug, os.path.join(debug_dir, 'spen_debug_tensors.pt'))
                if show_viz:
                    fg_sim_maps.append(aux_attr['raw_local_sims'])
            # print(f"Time for fg: {time.time() - start_time}")
            '''
            步骤 8：预测结果上采样。 将背景得分和各类别前景得分在通道维度拼接作为预测张量 pred，使用双线性插值将其放大回原始输入图像的尺寸 
            '''
            pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
            interpolate_mode = 'bilinear'
            outputs.append(F.interpolate(
                pred, size=img_size, mode=interpolate_mode))
            '''
            步骤 9：计算对齐损失（仅训练期）。 如果模型配置中开启了 align 且处于训练模式，则调用 alignLoss 函数，
            利用生成的预测图计算原型对齐损失，并累加到总损失中。
            '''
            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi
        '''
        步骤 10：输出格式化。 将多维预测结果展平整理为规范输出格式，计算平均对齐损失，
        并打包相关的相似度可视化图和查询特征一并返回给调用者。
        '''
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        grid_shape = output.shape[2:]
        if self.config["cls_name"] == 'grid_proto_3d':
            grid_shape = output.shape[2:]
        output = output.view(-1, *grid_shape)
        assign_maps = torch.stack(assign_maps, dim=1) if show_viz else None
        bg_sim_maps = torch.stack(bg_sim_maps, dim=1) if show_viz else None
        fg_sim_maps = torch.stack(fg_sim_maps, dim=1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps, proto_grid, supp_fts, qry_fts

    #该函数执行模型的核心前向传播，处理支持集和查询集数据以生成原型并计算相似度，最终输出分割预测结果张量及对齐损失
    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  # 1 x  N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        # FIXME: fix this in future we here make a stronger assumption that a positive class must be there to avoid undersegmentation/ lazyness
        skip_ways = []

        # added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(
            2)  # added to nway(1) and nb(1)
        # end of added part

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]
                img_fts = supp_fts[way: way + 1, shot: shot + 1]
                size = img_fts.shape[-2:]
                mode = 'bilinear'
                if self.config["cls_name"] == 'grid_proto_3d':
                    size = img_fts.shape[-3:]
                    mode = 'trilinear'
                qry_pred_fg_msk = F.interpolate(
                    binary_masks[way + 1].float(), size=size, mode=mode)  # [1 (way), n (shot), h, w]

                # background
                qry_pred_bg_msk = F.interpolate(
                    binary_masks[0].float(), size=size, mode=mode)  # 1, n, h ,w
                scores = []

                bg_mode = BG_PROT_MODE
                _raw_score_bg, _, _, _ = self.cls_unit(
                    qry=img_fts, sup_x=qry_fts, sup_y=qry_pred_bg_msk.unsqueeze(-3), mode=bg_mode, thresh=BG_THRESH)

                scores.append(_raw_score_bg)
                if self.config["cls_name"] == 'grid_proto_3d':
                    fg_mode = FG_PROT_MODE if F.avg_pool3d(qry_pred_fg_msk, 4).max(
                    ) >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'
                else:
                    fg_mode = FG_PROT_MODE if F.avg_pool2d(qry_pred_fg_msk, 4).max(
                    ) >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'
                _raw_score_fg, _, _, _ = self.cls_unit(
                    qry=img_fts,
                    sup_x=qry_fts,
                    sup_y=qry_pred_fg_msk.unsqueeze(2),
                    mode=fg_mode,
                    thresh=FG_THRESH,
                    background_score=(
                        _raw_score_bg
                        if self.config.get('cls_name', '').startswith('dense_fg_')
                        else None
                    ),
                )
                scores.append(_raw_score_fg)

                supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
                size = fore_mask.shape[-2:]
                if self.config["cls_name"] == 'grid_proto_3d':
                    size = fore_mask.shape[-3:]
                supp_pred = F.interpolate(supp_pred, size=size, mode=mode)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss.append(F.cross_entropy(
                    supp_pred.float(), supp_label[None, ...], ignore_index=255) / n_shots / n_ways)

        return torch.sum(torch.stack(loss))

    #该函数处理教师和学生的分类Token，使用 Sinkhorn-Knopp 算法对教师输出进行中心化后计算交叉熵蒸馏损失
    def dino_cls_loss(self, teacher_cls_tokens, student_cls_tokens):
        cls_loss_weight = 0.1
        student_temp = 1
        teacher_cls_tokens = self.sinkhorn_knopp_teacher(teacher_cls_tokens)
        lsm = F.log_softmax(student_cls_tokens / student_temp, dim=-1)
        cls_loss = torch.sum(teacher_cls_tokens * lsm, dim=-1)

        return -cls_loss.mean() * cls_loss_weight

    #该函数对输入的教师网络输出特征 teacher_output 进行 Sinkhorn-Knopp 迭代计算，生成平滑且归一化的软标签矩阵。
    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp=1, n_iterations=3):
        teacher_output = teacher_output.float()
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        # Q is K-by-B for consistency with notations from our paper
        Q = torch.exp(teacher_output / teacher_temp).t()
        # B = Q.shape[1] * world_size # number of samples to assign
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    #该函数对比学生特征 features 与教师特征 masked_features，针对掩码区域 masks 计算 Patch 级别的知识蒸馏损失。
    def dino_patch_loss(self, features, masked_features, masks):
        # for both supp and query features perform the patch wise loss
        loss = 0.0
        weight = 0.1
        B = features.shape[0]
        for (f, mf, mask) in zip(features, masked_features, masks):
            # TODO sinkhorn knopp center features
            f = f[mask]
            f = self.sinkhorn_knopp_teacher(f)
            mf = mf[mask]
            loss += torch.sum(f * F.log_softmax(mf / 1,
                              dim=-1), dim=-1) / mask.sum()

        return -loss.sum() * weight / B
