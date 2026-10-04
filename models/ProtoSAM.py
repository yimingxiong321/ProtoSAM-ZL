import os
import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from models.grid_proto_fewshot import FewShotSeg
from models.segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
    SamPredictor,
)
from models.SamWrapper import SamWrapper
from util.utils import cca, get_connected_components, rotate_tensor_no_crop, reverse_tensor, get_confidence_from_logits
from util.lora import inject_trainable_lora
from util.sam_prompt_utils import build_soft_prompt_mask_input
from util.candidate_audit import audit_candidate_proposals
from util.mnn_prompt import select_mnn_prompt_points
from models.presence_gate import SupportConditionedPresenceScorer
from models.segment_anything.utils.transforms import ResizeLongestSide
import cv2
import time
from abc import ABC, abstractmethod

CONF_MODE="conf"
CENTROID_MODE="centroid"
BOTH_MODE="both"
POINT_MODES=(CONF_MODE, CENTROID_MODE, BOTH_MODE)

TYPE_ALPNET="alpnet"
TYPE_SAM="sam"

def plot_connected_components(cca_output, original_image, confidences:dict=None, title="debug/connected_components.png"):
    num_labels, labels, stats, centroids = cca_output
    # Create an output image with random colors for each component
    output_image = np.zeros((labels.shape[0], labels.shape[1], 3), np.uint8)
    for label in range(1, num_labels):  # Start from 1 to skip the background
        mask = labels == label
        output_image[mask] = np.random.randint(0, 255, size=3)

    # Plotting the original and the colored components image
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(original_image), plt.title('Original Image')
    plt.subplot(122), plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)), plt.title('Connected Components')
    if confidences is not None:
        # Plot the axes color chart with the confidences, use the same colors as the connected components
        plt.subplot(122)
        scatter = plt.scatter(centroids[:, 0], centroids[:, 1], c=list(confidences.values()), cmap='jet')
        plt.colorbar(scatter)

    plt.savefig(title)
    plt.close()

#1.分割网络的输入输出类都必须包含 set_query_images 这样的通用方法
class SegmentationInput(ABC):
    @abstractmethod
    def set_query_images(self, query_images):
        pass
    
    def to(self, device):
        pass
    
class SegmentationOutput(ABC):
    @abstractmethod
    def get_prediction(self):
        pass

#2. 包装底层模型，统一测试/训练调用     
class ModelWrapper(ABC):
    def __init__(self, model):
        self.model = model
        
    def __call__(self, input_data: SegmentationInput)->SegmentationOutput:
        pass

    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train() 
    
    def parameters(self):
        pass
        
#3. ALPNetInput、ALPNetOutput 与 ALPNetWrapper：这些模块专门针对 ALPNet（一个小样本分割模型）把数据打包为模型输入形式
class ALPNetInput(SegmentationInput): # for alpnet
    def __init__(self, support_images:list, support_labels:list, query_images:torch.Tensor, isval, val_wsize, show_viz=False, supp_fts=None, support_weights=None):
        self.supp_imgs = [support_images]
        self.fore_mask = [support_labels]
        self.back_mask = [[1 - sup_labels for sup_labels in support_labels]]
        self.qry_imgs = [query_images]
        self.isval = isval #这四个变量是什么含义，模型输入需要这些数据吗
        self.val_wsize = val_wsize
        self.show_viz = show_viz
        self.supp_fts = supp_fts
        # QSPA: softmax weights over top-K supports. None keeps original 1-shot / max-shot path.
        self.support_weights = support_weights
        
    def set_query_images(self, query_images):
        self.qry_imgs = [query_images]
        
    def to(self, device):
        self.supp_imgs = [[supp_img.to(device) for way in self.supp_imgs for supp_img in way]]
        self.fore_mask = [[fore_mask.to(device) for way in self.fore_mask for fore_mask in way]]
        self.back_mask = [[back_mask.to(device) for way in self.back_mask for back_mask in way]]
        self.qry_imgs = [qry_img.to(device) for qry_img in self.qry_imgs]
        if self.supp_fts is not None:
            self.supp_fts = self.supp_fts.to(device)
        if self.support_weights is not None and torch.is_tensor(self.support_weights):
            self.support_weights = self.support_weights.to(device)

class ALPNetOutput(SegmentationOutput):
    def __init__(self, pred, align_loss, sim_maps, assign_maps, proto_grid, supp_fts, qry_fts):
        self.pred = pred #这些指标是什么含义
        self.align_loss = align_loss
        self.sim_maps = sim_maps
        self.assign_maps = assign_maps
        self.proto_grid = proto_grid
        self.supp_fts = supp_fts
        self.qry_fts = qry_fts
        
    def get_prediction(self):
        return self.pred


class ALPNetWrapper(ModelWrapper):
    def __init__(self, model: FewShotSeg):
        super().__init__(model)
        
    def __call__(self, input_data: ALPNetInput):
        output = self.model(**input_data.__dict__)
        output = ALPNetOutput(*output)
        self.last_input = input_data
        self.last_output = output
        return output.pred
    
    def parameters(self):
        return self.model.encoder.parameters()
    
    def train(self):
        self.model.encoder.train()

#4. 专门针对 SAM（Segment Anything Model）。它们负责将输入图像和标签的维度、张量类型以及数值范围标准化
# （例如缩放到 0-255 并转换为 numpy 格式），以满足 SAM 严格的输入要求
class SAMWrapperInput(SegmentationInput):
    def __init__(self, image, image_labels):
        self.image = image
        self.image_labels = image_labels
        
    def set_query_images(self, query_images):
        B, C, H, W = query_images.shape
        if isinstance(query_images, torch.Tensor):
            query_images = query_images.cpu().detach().numpy()
        assert B == 1, "batch size must be 1"
        query_images = (query_images - query_images.min()) / (query_images.max() - query_images.min()) * 255
        query_images = query_images.astype(np.uint8)
        self.image = np.transpose(query_images[0], (1, 2, 0))
        
    def to(self, device):
        pass

        
class SamWrapperWrapper(ModelWrapper):
    def __init__(self, model:SamWrapper):
        super().__init__(model)
        
    def __call__(self, input_data: SAMWrapperInput):
        pred = self.model(**input_data.__dict__)
        # make pred look like logits
        pred = torch.tensor(pred).float()[None, None, ...]
        pred = torch.cat([1-pred, pred], dim=1)
        return pred

    def to(self, device):
        self.model.sam.to(device)
    
#5. 根据（TYPE_ALPNET 或 TYPE_SAM）调用对应的模型输入处理类进行数据处理，返回对应模型可以接受的数据形式（对象）
class InputFactory(ABC):
    @staticmethod
    def create_input(input_type, query_image, support_images=None, support_labels=None, isval=False, val_wsize=None, show_viz=False, supp_fts=None, original_sz=None, img_sz=None, gts=None, support_weights=None):
        
        if input_type == TYPE_ALPNET:
            return ALPNetInput(support_images, support_labels, query_image, isval, val_wsize, show_viz, supp_fts, support_weights=support_weights)
        elif input_type == TYPE_SAM:
            qimg = np.array(query_image.detach().cpu())
            B,C,H,W = qimg.shape
            assert B == 1, "batch size must be 1"
            gts = np.array(gts.detach().cpu()).astype(np.uint8).reshape(H,W) #gts是粗分割概率图吗
            assert np.unique(gts).shape[0] <= 2, "support labels must be binary"
            gts[gts > 0] = 1
            qimg = qimg.reshape(H,W,C)
            qimg = (qimg - qimg.min()) / (qimg.max() - qimg.min()) * 255
            qimg = qimg.astype(np.uint8)
            return SAMWrapperInput(qimg, gts)
        else:
            raise ValueError(f"input_type not supported")
 
#DINOv2 backbone → ALPNet 粗分割 → SAM 精修
class ProtoSAM(nn.Module):

    # 作用：初始化 ProtoSAM 模块，绑定传入的粗分割模型，加载 SAM 权重，并设定后续生成 SAM 提示词（点、框、掩码）的各种模式和参数。
    def __init__(self, image_size, coarse_segmentation_model:ModelWrapper, sam_pretrained_path="pretrained_model/sam_default.pth", num_points_for_sam=1, use_points=True, use_bbox=False, use_mask=False, debug=False, use_cca=False, point_mode=CONF_MODE, use_sam_trans=True, coarse_pred_only=False, alpnet_image_size=None, use_neg_points=False, ablation_mode='none', ablation_fixes=None, candidate_audit=False, candidate_audit_thresholds=(0.3, 0.4, 0.5), sam3_text_prompt="polyp", ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.coarse_segmentation_model = coarse_segmentation_model
        self.sam3_text_prompt = sam3_text_prompt
        self.get_sam(sam_pretrained_path, use_sam_trans) #use_sam_trans的意思是配置图像变换
        self.num_points_for_sam = num_points_for_sam
        self.use_points = use_points
        self.use_bbox = use_bbox # if False then uses points
        self.use_mask = use_mask
        self.use_neg_points = use_neg_points
        assert self.use_bbox or self.use_points or self.use_mask, "must use at least one of bbox, points, or mask"
        self.use_cca = use_cca
        self.point_mode = point_mode
        if self.point_mode not in POINT_MODES:
            raise ValueError(f"point mode must be one of {POINT_MODES}")
        self.debug=debug
        self.coarse_pred_only = coarse_pred_only
        self.ablation_mode = ablation_mode
        self.ablation_fixes = ablation_fixes or []
        self.candidate_audit = bool(candidate_audit)
        self.candidate_audit_thresholds = tuple(candidate_audit_thresholds)
        # P0/P1 is score-only: it never rejects a query and never changes the
        # coarse prediction.  Threshold selection is deliberately left to a
        # patient-wise validation/calibration split.
        self.presence_scorer = (
            SupportConditionedPresenceScorer()
            if self.ablation_mode == "presence_score" or self.candidate_audit
            else None
        )
        self.last_presence_scores = None
        self.last_candidate_proposals = None
        self.last_proposal_slices = None

    def eval(self):
        super().eval()
        # ALPNetWrapper is not an nn.Module, so nn.Module.eval() does not reach it.
        if hasattr(self.coarse_segmentation_model, "eval"):
            self.coarse_segmentation_model.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self.coarse_segmentation_model, "train"):
            self.coarse_segmentation_model.train()
        return self

    def compute_presence_scores(self, foreground_probability):
        """Reuse the coarse wrapper's cached DINOv2 tensors without re-encoding."""
        if self.presence_scorer is None:
            return None
        coarse_wrapper = self.coarse_segmentation_model
        coarse_output = getattr(coarse_wrapper, "last_output", None)
        coarse_input = getattr(coarse_wrapper, "last_input", None)
        if coarse_output is None or coarse_input is None:
            raise RuntimeError(
                "presence_score requires a coarse wrapper that exposes last_input/last_output"
            )
        scores = self.presence_scorer(
            support_features=coarse_output.supp_fts,
            query_features=coarse_output.qry_fts,
            support_mask=coarse_input.fore_mask[0][0][0],
            foreground_probability=foreground_probability,
        )
        self.last_presence_scores = scores.to_dict()
        return scores

    # 作用：根据权重路径加载预训练的 SAM 模型及其专用推理器 (SamPredictor)，并可选择性地配置 SAM 特有的图像缩放与归一化变换 (ResizeLongestSide)。     
    def get_sam(self, checkpoint_path, use_sam_trans):
        from models.sam3_grounding import is_sam3_checkpoint, load_sam3_grounding_runtime
        from models.sam3_predictor import _SamStub

        self.sam3_runtime = None
        if is_sam3_checkpoint(checkpoint_path):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            text_prompt = getattr(self, "sam3_text_prompt", "polyp")
            self.sam3_runtime = load_sam3_grounding_runtime(
                checkpoint_path=checkpoint_path,
                device=device,
                text_prompt=text_prompt,
            )
            self.predictor = None
            self.sam = _SamStub()
            self._sam3_backend = True
        else:
            self._sam3_backend = False
            model_type = "vit_b"  # TODO make generic?
            if "vit_h" in checkpoint_path:
                model_type = "vit_h"
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).eval()
            self.predictor = SamPredictor(self.sam)
            self.sam.requires_grad_(False)
        if use_sam_trans:
            sam_trans = ResizeLongestSide(self.sam.image_encoder.img_size)
            sam_trans.pixel_mean = torch.tensor([0, 0, 0]).view(3, 1, 1)
            sam_trans.pixel_std = torch.tensor([1, 1, 1]).view(3, 1, 1)
        else:
            sam_trans = None

        self.sam_trans = sam_trans

    # 作用：接收二维的二值化预测图 (pred)，计算并返回一个能够包围图中所有前景像素的全局边界框的四个角点坐标。    
    def get_bbox(self, pred):
        '''
        pred tensor of shape (H, W) where 1 represents foreground and 0 represents background
        returns a list of 2d points representing the bbox
        ''' 
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred)
        # get the indices of the foreground points
        indices = torch.nonzero(pred)
        # get the min and max of the indices
        min_x = indices[:, 1].min()
        max_x = indices[:, 1].max()
        min_y = indices[:, 0].min()
        max_y = indices[:, 0].max()
        # get the bbox
        bbox = [[min_y, min_x], [min_y, max_x], [max_y, max_x], [max_y, min_x]]
        
         
        return bbox  

    #粗分割模型会输出一个全局的预测结果，需要在外部做一次连通域分析（CCA）分解为多个pred（连通块） 
    # 作用：遍历连通域分析 (CCA) 输出的各个独立连通块，为每一个前景连通块计算并返回 [x_min, y_min, x_max, y_max] 格式的边界框列表。
    def get_bbox_per_cc(self, conn_components):
        """
        conn_components: output of cca function
        return list of bboxes per connected component, each bbox is a list of 2d points
        """
        bboxes = []
        for i in range(1, conn_components[0]):
            # get the indices of the foreground points
            indices = torch.nonzero(torch.tensor(conn_components[1] == i))
            # get the min and max of the indices
            min_x = indices[:, 1].min()
            max_x = indices[:, 1].max()
            min_y = indices[:, 0].min()
            max_y = indices[:, 0].max()
            # get the bbox
            # bbox = [[min_y, min_x], [min_y, max_x], [max_y, max_x], [max_y, min_x]]
            # bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            # bbox should be in a XYXY format
            bbox = [min_x, min_y, max_x, max_y]
            bboxes.append(bbox)

        bboxes = np.array(bboxes)
        return bboxes

    # 作用：根据前景概率图 (output_p_fg) 和二值预测图 (pred)，提取出前 k 个置信度最高（概率最大）的像素点坐标及其概率值，用作 SAM 的高质量正样本点提示。
    def get_most_conf_points(self, output_p_fg, pred, k):
        '''
        get the k most confident points from pred

        output_p: 3d tensor of shape (H, W)  连续的“概率热力图” 在这里，粗分割模型输入的是单张查询图像，所以 Batch=1；
        这是一个二分类任务（区分前景和背景），所以 Channel=2。因此，它初始生成的是一个形状为 (1, 2, H, W) 的 4D 张量。
        经过 Softmax 计算得到概率图 output_p 代码在 dim=1（也就是 Channel 维度）上做了 Softmax 激活。这一步将网络输出的原始数值转换成了概率值。
        其中，通道 0 存放的是每个像素属于“背景”的概率，通道 1 存放的是属于“前景”的概率。此时 output_p 依然是 (1, 2, H, W) 的 4D 张量。
        pred: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
        '''
        # Create a mask where pred is 1
        mask = pred.bool()

        # Apply the mask to output_p_fg
        masked_output_p_fg = output_p_fg[mask]
        if masked_output_p_fg.numel() == 0:
            return None, None
        # Get the top k probabilities and their indices
        confidences, indices = torch.topk(masked_output_p_fg, k)

        # 返回所有为 True 的像素索引，形状为 (N, 2)，也就是 [y（行）, x（列）]
        locations = torch.nonzero(mask)[indices]
        # 将坐标顺序调换成了图像处理中常用的 [x, y] 格式
        locations = locations[:, [1, 0]]
        # convert locations to list of lists
        # points = [loc.tolist() for loc in locations]
        
        return locations.numpy(), [float(conf.item()) for conf in confidences]
    
    # 作用：调试用可视化函数，将提取到的高置信度提示点、边界框以及粗分割预测结果叠加绘制在原始输入图像上并保存。
    def plot_most_conf_points(self, points, confidences, pred, image, bboxes=None, title=None):
        '''
        points: np array of shape (N, 2) where each row is a point in xy format
        pred: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
        image: 2d tensor of shape (H,W) representing the image
        bbox: list or np array of shape (N, 4) where each row is a bbox in xyxy format
        '''
        warnings.filterwarnings('ignore', category=UserWarning)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().detach().numpy()
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        if title is None:
            title="debug/most_conf_points.png"
            
        fig = plt.figure()
        image = (image - image.min()) / (image.max() - image.min())
        plt.imshow(image)
        plt.imshow(pred, alpha=0.5)
        for i, point in enumerate(points):
            plt.scatter(point[0][0], point[0][1], cmap='viridis', marker='*', c='red')
            if confidences is not None:
                plt.text(point[0], point[1], f"{confidences[i]:.3f}", fontsize=12, color='red')
        # assume points is a list of lists
        if bboxes is not None:
            for bbox in bboxes:
                if bbox is None:
                    continue
                bbox = np.array(bbox)
                # plt.scatter(bbox[:, 1], bbox[:, 0], c='red')
                # plot a line connecting the points
                box = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
                box = np.vstack([box, box[0]])
                plt.plot(box[:, 0], box[:, 1], c='green')
        plt.colorbar()
        fig.savefig(title)
        plt.close(fig)

    # 作用：调试用可视化函数，将 SAM 模型最终预测出的精细掩码 (masks)、置信度分数以及所用的提示条件叠加绘制在原图上。    
    def plot_sam_preds(self, masks, scores, image, input_point, input_label, input_box=None):
        if len(image.shape) == 3:
            image = image.permute(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            if input_point is not None:
                show_points(input_point, input_label, plt.gca())
            if input_box is not None:
                show_box(input_box, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            # plt.axis('off')
            plt.savefig(f'debug/sam_mask_{i+1}.png')
            plt.close()
            if i > 5:
                break

    # 作用：解析连通域数据，结合概率图为每个连通块提取用于 SAM 推理的正样本点（最高置信度点或几何中心），并根据配置计算背景区域的负样本点。    
    def get_sam_input_points(self, conn_components, output_p, get_neg_points=False, l=1): 
        """
        args:
        conn_components: output of cca function
        output_p: 3d tensor of shape (1, 2, H, W)
        get_neg_points: bool, if True then return the negative points
        l: int, number of negative points to get
        """
        sam_input_points = []
        sam_neg_points = []

        '''
        这里的 [0, 1] 就是具体的降维操作：  
        0：取出 Batch 维度中的第 0 个数据（由于 batch size 是 1，这就剥离了 Batch 维度）。  
        1：取出 Channel 维度中的第 1 个通道（也就是代表前景概率的通道，剥离了 Channel 维度）。
        经过这两次切片，剥离了前两个维度后，剩下的 fg_p 就变成了一个纯粹的、尺寸为 (H, W) 的 2D 概率矩阵。(前景概率图)
        同时代码提取了背景图：bg_p = output_p[0, 0].detach().cpu()（通道 0）用于后续可能的负样本点提取。  
        '''
        fg_p = output_p[0, 1].detach().cpu()
        
        if get_neg_points:
            # get global negative points
            bg_p = output_p[0, 0].detach().cpu()
            bg_p[bg_p < 0.95] = 0
            bg_pred = torch.where(bg_p > 0, 1, 0)
            glob_neg_points, _ = self.get_most_conf_points(bg_p, bg_pred, 1)
            if self.debug:
                # plot the bg_p as a heatmap
                plt.figure()
                plt.imshow(bg_p)
                plt.colorbar()
                plt.savefig('debug/bg_p_heatmap.png')
                plt.close()
        
        for i, cc_id in enumerate(np.unique(conn_components[1])):
            # get self.num_points_for_sam most confident points from pred
            if cc_id == 0:
                continue  # skip background
            pred = torch.tensor(conn_components[1] == cc_id).float()

            if self.point_mode == CONF_MODE:
                points, confidences = self.get_most_conf_points(fg_p, pred, self.num_points_for_sam)  # (N, 2)
            elif self.point_mode == CENTROID_MODE:
                points = conn_components[3][cc_id][None, :]  # (1, 2)
                confidences = [1 for _ in range(len(points))]
            elif self.point_mode == BOTH_MODE:
                points, confidences = self.get_most_conf_points(fg_p, pred, self.num_points_for_sam)
                point = conn_components[3][cc_id][None, :]
                points = np.vstack([points, point])  # (N+1, 2)
                confidences.append(1)
            else:
                raise NotImplementedError(f"point mode {self.point_mode} not implemented")
            sam_input_points.append(np.array(points))
            
            if get_neg_points:
                pred_uint8 = (pred.numpy() * 255).astype(np.uint8)

                # Dilate the mask to expand it
                kernel_size = 3  # Size of the dilation kernel, adjust accordingly
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                dilation_iterations = 10  # Number of times dilation is applied, adjust as needed
                dilated_mask = cv2.dilate(pred_uint8, kernel, iterations=dilation_iterations)

                # Subtract the original mask from the dilated mask
                # This will give a boundary that is only outside the original mask
                outside_boundary = dilated_mask - pred_uint8

                # Convert back to torch tensor and normalize
                boundary = torch.tensor(outside_boundary).float() / 255
                try:
                    bg_p = output_p[0, 0].detach().cpu()
                    neg_points, neg_confidences = self.get_most_conf_points(bg_p, boundary, l)
                except RuntimeError as e:
                    # make each point (None, None)
                    neg_points = None
                # append global negative points to the negative points
                if neg_points is not None and glob_neg_points is not None:
                    neg_points = np.vstack([neg_points, glob_neg_points])
                else:
                    neg_points = glob_neg_points if neg_points is None else neg_points
                if self.debug and neg_points is not None:
                    # draw an image with 2 subplots, one is the pred and the other is the boundary
                    plt.figure()
                    plt.subplot(121)
                    plt.imshow(pred)
                    plt.imshow(boundary, alpha=0.5)
                    # plot the neg points
                    plt.scatter(neg_points[:, 0], neg_points[:, 1], cmap='viridis', marker='*', c='red')
                    plt.subplot(122)
                    plt.imshow(pred)
                    plt.scatter(neg_points[:, 0], neg_points[:, 1], cmap='viridis', marker='*', c='red')
                    plt.savefig('debug/pred_and_boundary.png')
                    plt.close()
                sam_neg_points.append(neg_points)
            else:
                # create a list of None same shape as points
                sam_neg_points = [None for _ in range(len(sam_input_points))]

        sam_input_labels = np.array([l+1 for l, cc_points in enumerate(sam_input_points) for _ in range(len(cc_points))])
        sam_input_points = np.stack(sam_input_points)  # should be of shape (num_connected_components, num_points_for_sam, 2)
        # if get_neg_points:
        sam_neg_input_points = np.stack(sam_neg_points) if sam_neg_points is not None else None
        if sam_neg_input_points is not None:
            sam_neg_input_points = sam_neg_points
            sam_neg_input_labels = np.array([0] * len(sam_neg_input_points) )
        else:
            sam_neg_input_points = None
            sam_neg_input_labels = None

        return sam_input_points, sam_input_labels, sam_neg_input_points, sam_neg_input_labels

    def replace_confidence_point_with_mnn(self, sam_input_points, conn_components):
        """Replace one confidence point per component with a reciprocal match."""
        coarse_wrapper = self.coarse_segmentation_model
        coarse_output = getattr(coarse_wrapper, "last_output", None)
        coarse_input = getattr(coarse_wrapper, "last_input", None)
        if coarse_output is None or coarse_input is None:
            return sam_input_points
        try:
            support_features = coarse_output.supp_fts[0, 0, 0]
            query_features = coarse_output.qry_fts[0, 0]
            support_mask = coarse_input.fore_mask[0][0][0]
        except (AttributeError, IndexError, TypeError):
            return sam_input_points

        component_labels = np.asarray(conn_components[1])
        points, _ = select_mnn_prompt_points(
            support_features=support_features,
            query_features=query_features,
            support_mask=support_mask,
            query_candidate_mask=component_labels > 0,
            max_points=max(1, len(np.unique(component_labels)) - 1),
        )
        if len(points) == 0:
            return sam_input_points

        foreground_ids = [label for label in np.unique(component_labels) if label != 0]
        id_to_array_index = {label: index for index, label in enumerate(foreground_ids)}
        replaced = set()
        for point in points:
            x = int(np.clip(round(float(point[0])), 0, component_labels.shape[1] - 1))
            y = int(np.clip(round(float(point[1])), 0, component_labels.shape[0] - 1))
            component_id = component_labels[y, x]
            if component_id == 0 or component_id in replaced:
                continue
            array_index = id_to_array_index.get(component_id)
            if array_index is None or sam_input_points[array_index] is None:
                continue
            sam_input_points[array_index][0] = point
            replaced.add(component_id)
        return sam_input_points

    # 作用：遍历连通域数据，将每一个独立的前景连通块转换为单独的二值掩码张量，以作为 SAM 的低分辨率 Mask 提示词。
    def get_sam_input_mask(self, conn_components):
        sam_input_masks = []
        sam_input_mask_lables = []
        for i, cc_id in enumerate(np.unique(conn_components[1])):
            # get self.num_points_for_sam most confident points from pred
            if cc_id == 0:
                continue
            pred = torch.tensor(conn_components[1] == cc_id).float()
            sam_input_masks.append(pred)
            sam_input_mask_lables.append(cc_id)

        sam_input_masks = np.stack(sam_input_masks)
        sam_input_mask_lables = np.array(sam_input_mask_lables)
        
        return sam_input_masks, sam_input_mask_lables

    # 作用：将掩码提示词转换为 SAM 需要的具体数值格式，喂入预测器生成高精度的分割掩码，并返回得分最高的预测结果。
    def predict_w_masks(self, sam_input_masks, qry_img, original_size):
        masks = []
        scores = []
        for in_mask in sam_input_masks:
            in_mask = cv2.resize(in_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            in_mask[in_mask == 1] = 10
            in_mask[in_mask == 0] = -8
            assert qry_img.max() <= 255 and qry_img.min() >= 0 and qry_img.dtype == np.uint8   
            self.predictor.set_image(qry_img)
            mask, score, _ = self.predictor.predict(
                mask_input=in_mask[None, ...].astype(np.float32),
                multimask_output=True)
            # get max index from score
            if self.debug:
                # plot each channel of mask
                fig, ax = plt.subplots(1, 4, figsize=(15, 5))
                for i in range(mask.shape[0]):
                    ax[i].imshow(qry_img)
                    ax[i].imshow(mask[i], alpha=0.5)
                    ax[i].set_title(f"Mask {i+1}, Score: {score[i]:.3f}", fontsize=18)
                    # ax[i].axis('off')
                ax[-1].imshow(cv2.resize(in_mask, original_size, interpolation=cv2.INTER_NEAREST))
                fig.savefig(f'debug/sam_mask_from_mask_prompts.png')
                plt.close(fig)
                           
                        
            max_index = score.argmax()
            masks.append(mask[max_index])
            scores.append(score[max_index])
        
        return masks, scores

    def predict_w_bbox_sam3(
        self,
        bboxes,
        pil_rgb,
        prompt_hw,
        orig_hw,
        sam_input_points=None,
        return_logits=False,
    ):
        """SAM3 T+I refinement: text + coarse bbox + optional Conf/Cent points (same as SAM1)."""
        from models.sam3_grounding import predict_grounding_masks_for_boxes

        if self.sam3_runtime is None:
            raise RuntimeError("sam3_runtime is not loaded")
        points_seq = None
        _sam3_pts = os.environ.get("SAM3_USE_COARSE_POINTS", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        if _sam3_pts and self.use_points and sam_input_points is not None:
            points_seq = [np.asarray(p) if p is not None else None for p in sam_input_points]
        return predict_grounding_masks_for_boxes(
            self.sam3_runtime,
            pil_rgb,
            bboxes,
            tuple(prompt_hw),
            tuple(orig_hw),
            points_prompt_space=points_seq,
            return_logits=return_logits,
        )

    # 作用：将正负样本点和/或边界框作为组合提示词喂入 SAM 预测器，针对输入图像 (qry_img) 推理出精修后的分割掩码与对应得分。
    def predict_w_points_bbox(self, sam_input_points, bboxes, sam_neg_input_points, qry_img, pred, return_logits=False, sam_mask_input=None):
        masks, scores = [], []
        self.predictor.set_image(qry_img)
        # if sam_input_points is None:
        #     sam_input_points = [None for _ in range(len(bboxes))]
        for point, bbox_xyxy, neg_point in zip(sam_input_points, bboxes, sam_neg_input_points): 
            assert qry_img.max() <= 255 and qry_img.min() >= 0 and qry_img.dtype == np.uint8   
            points = point
            point_labels = np.array([1] * len(point)) if point is not None else None
            if self.use_neg_points:
                neg_points = [npoint for npoint in neg_point if None not in npoint] 
                points = np.vstack([point, *neg_points])
                point_labels = np.array([1] * len(point) + [0] * len(neg_points))
            if self.debug: 
                self.plot_most_conf_points(points[:, None, ...], None, pred, qry_img, bboxes=bbox_xyxy[None,...] if bbox_xyxy is not None else None, title="debug/pos_neg_points.png") # TODO add plots for all points not just the first set of points
            mask, score, _ = self.predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                # box=bbox_xyxy[None, :] if bbox_xyxy is not None else None,
                box = bbox_xyxy if bbox_xyxy is not None else None,
                mask_input=sam_mask_input if sam_mask_input is not None else None,
                return_logits=return_logits,
                multimask_output=False if (self.use_cca and 'multimask_score' not in self.ablation_fixes) else True
            )
            # best_pred_idx = np.argmax(score)
            best_pred_idx = np.argmax(score) if 'multimask_score' in self.ablation_fixes else 0
            masks.append(mask[best_pred_idx])
            scores.append(score[best_pred_idx])
        
        if self.debug:
            # pass
            self.plot_sam_preds(mask, score, qry_img[...,0], points.reshape(-1,2) if sam_input_points is not None else None, point_labels, input_box=bbox_xyxy if bbox_xyxy is not None else None)

        return masks, scores
    
    
    def forward(self, query_image, coarse_model_input, degrees_rotate=0, gt_mask=None):
        """
        query_image: 3d tensor of shape (1, 3, H, W)
        images should be normalized with mean and std but not to [0, 1]?
        """
        self.last_presence_scores = None
        self.last_candidate_proposals = None
        self.last_proposal_slices = None
        original_size = query_image.shape[-2:] #提取输入图像的原始尺寸

        # 步骤 1：图像预处理与旋转
        # 将输入图像进行可选的旋转操作（如果 degrees_rotate != 0），用于数据增强或测试时增强 (TTA)。
        start_time = time.time()
        rotated_img, (rot_h, rot_w) = rotate_tensor_no_crop(query_image, degrees_rotate)
        # print(f"rotating query image took {time.time() - start_time} seconds")

        # 步骤 2：粗分割模型推理
        # 将处理后的图像喂入 coarse_segmentation_model（如 ALPNet）获取粗糙的初始预测对数 (Logits)。
        start_time = time.time()
        # MULTI_SUPPORT_PATCHED: support list averaging 【为什么有两条分支】
        '''
        coarse_model_input 可能有两种不同的类型：

            单个对象__（`ALPNetInput` 实例）：常规情况，只有一个支持集（support set），直接推理一次。
            一个列表__（`list`，里面装多个 `ALPNetInput`）：这是 `multi_support`（多支持集）消融实验模式。
            此时有多个支持集，每个支持集单独推理一次，最后把多个结果平均。

            所以 `isinstance(coarse_model_input, list)` 就是判断当前是不是多支持集模式，从而走不同的处理逻辑。

        coarse_model_input 是怎么来的？
            - `TYPE_ALPNET` → 返回 `ALPNetInput` 对象，它把支持集图像、支持集标签、查询图像等打包成一个对象。
            - `TYPE_SAM` → 返回 `SAMWrapperInput` 对象。
            通过调用方__（如 `validation_protosam.py`）通过 `InputFactory.create_input(...)` 创建的。
            `coarse_model_input` 是 `ALPNetInput` 对象，这个对象封装了查询图像和支持集。
            这个对象被用来调用 `self.coarse_segmentation_model`，即调用 `ALPNet` 模型进行推理。
        '''
        if isinstance(coarse_model_input, list):
            _all_logits = [] 
            for _cmi in coarse_model_input: #`[ALPNetInput, ALPNetInput, ...]` 的列表
                # 1. 挂载查询图像
                _cmi.set_query_images(rotated_img) #`cmi` 对象内部的查询图像就被替换成了 `rotated_img`
                # 2. 独立推理，获取该 Support 下的预测 logits
                _logits_rot = self.coarse_segmentation_model(_cmi) # 获取预测对数 
                # 3. 提前逆旋转（注意这里的差异！）
                '''
                即把 `ALPNetInput` 对象里的 `supp_imgs`、`fore_mask`、`back_mask`、`qry_imgs` 等属性解包，
                喂给底层的小样本分割模型 `FewShotSeg`，得到预测 logits
                '''
                #if degrees_rotate != 0:
                #    _logits = reverse_tensor(_logits_rot, rot_h, rot_w, -degrees_rotate)
                #else:
                #    _logits = _logits_rot
                _all_logits.append(_logits_rot)
            # 4. 融合：将所有 Support 的预测结果堆叠并沿第0维求平均
            output_logits = torch.stack(_all_logits, dim=0).mean(dim=0)
            # 5. 变量对齐
            output_logits_rot = output_logits  # for debug viz
        else:
            coarse_model_input.set_query_images(rotated_img)
            output_logits_rot = self.coarse_segmentation_model(coarse_model_input)
        # print(f"ALPNet took {time.time() - start_time} seconds")

        # 步骤 3：逆旋转与概率计算
        # 如果图像被旋转过，将粗分割的 Logits 逆向旋转回原始角度；随后通过 Softmax 激活函数得到每个像素属于前景/背景的概率图 (output_p) 和离散预测图 (pred)。
        if degrees_rotate != 0:
            start_time = time.time()
            output_logits = reverse_tensor(output_logits_rot, rot_h, rot_w, -degrees_rotate)
            # print(f"reversing rotated output_logits took {time.time() - start_time} seconds")
        else:
            output_logits = output_logits_rot
        
        # check if softmax is needed
        '''
        因为 `output_logits` 是一个 **4D 张量**，形状为 `(B, C, H, W)`，而 `dim=1` 正好是**通道（类别）维度**。

## 张量维度结构
在这个分割任务里，`C = 2`，即两个通道分别代表**背景**和**前景**的 logits。
## 为什么 softmax 用 dim=1
```python
output_p = output_logits.softmax(dim=1)
```
`softmax` 的作用是把一组数值变成概率（和为 1）。这里我们想让**每个像素**在"背景/前景"两个类别之间做归一化，所以要在**通道维度（dim=1）**上做 softmax。
- 对每个空间位置 `(h, w)`，取 `output_logits[:, :, h, w]`，即该像素在 2 个类别上的 logits
- 在这 2 个值上做 softmax，得到该像素属于背景/前景的概率
- 结果 `output_p` 形状仍是 `(B, 2, H, W)`，但每个像素的两个通道值加起来等于 1
**dim=1**，把同一像素的不同类别归一化。
## 为什么 argmax 用 dim=1
```python
pred = output_logits.argmax(dim=1)[0]
```
`argmax` 是取**最大值所在的索引**。我们要判断每个像素属于哪个类别，就要在**通道维度（dim=1）**上比较"背景 logits"和"前景 logits"谁更大：
- 如果背景通道值大 → 该像素判为背景（类别 0）
- 如果前景通道值大 → 该像素判为前景（类别 1）
`argmax(dim=1)` 返回形状为 `(B, H, W)` 的张量，每个位置的值是 0 或 1（即类别索引）。后面的 `[0]` 是去掉 Batch 维度，得到 `(H, W)` 的 2D 预测图。
## 一句话总结
`dim=1` 是**通道/类别维度**。softmax 在类别间做概率归一化，argmax 在类别间选最大者，两者都是对"每个像素属于哪个类别"这个问题的处理，所以都作用在 `dim=1` 上。
        ''' 
        output_p = output_logits.softmax(dim=1)
        # output_p = output_logits
        pred = output_logits.argmax(dim=1)[0] 

        if self.presence_scorer is not None:
            # Cached query features correspond to the rotated coarse-model
            # input.  All current experiments use zero rotation, but using the
            # rotated probability map here keeps the tensor frames consistent.
            presence_probability = (
                output_logits_rot.softmax(dim=1)[0, 1]
                if degrees_rotate != 0
                else output_p[0, 1]
            )
            self.compute_presence_scores(presence_probability)
            if self.candidate_audit:
                if gt_mask is None:
                    raise ValueError("candidate_audit requires gt_mask for offline diagnostics")
                proposals, slices = audit_candidate_proposals(
                    foreground_probability=output_p[0, 1],
                    likelihood_ratio_map=self.presence_scorer.last_likelihood_ratio_map,
                    gt_mask=gt_mask,
                    thresholds=self.candidate_audit_thresholds,
                )
                self.last_candidate_proposals = proposals
                self.last_proposal_slices = slices

        # 步骤 4：模式短路（仅需粗分割时）
        # 如果设置了 coarse_pred_only=True，则跳过后续所有 SAM 处理步骤，直接插值并返回粗分割模型的预测结果。
        if self.debug:
            _pred = np.array(output_logits.argmax(dim=1)[0].detach().cpu())
            plt.subplot(132)
            plt.imshow(query_image[0,0].detach().cpu())
            plt.imshow(_pred, alpha=0.5)
            plt.subplot(131)
            # plot heatmap of prob of being fg
            plt.imshow(output_p[0, 1].detach().cpu())
            # plot rotated query image and rotated pred
            output_p_rot = output_logits_rot.softmax(dim=1)
            _pred_rot = np.array(output_p_rot.argmax(dim=1)[0].detach().cpu())
            _pred_rot = F.interpolate(torch.tensor(_pred_rot).unsqueeze(0).unsqueeze(0).float(), size=original_size, mode='nearest')[0][0]
            plt.subplot(133)
            plt.imshow(rotated_img[0, 0].detach().cpu())
            plt.imshow(_pred_rot, alpha=0.5)
            plt.savefig('debug/coarse_pred.png')
            plt.close()
             
        if self.coarse_pred_only or self.ablation_mode in ("coarse_only", "presence_score"): 
            # interpolate to original size
            output_logits = F.interpolate(output_logits, size=original_size, mode='bilinear') if output_logits.shape[-2:] != original_size else output_logits
            pred = output_logits.argmax(dim=1)[0]
            conf = get_confidence_from_logits(output_logits) 
            '''
            “在所有被模型判定为前景的像素中，模型认为它们是前景的平均概率是多少？”
                如果返回值为 0.95：说明模型不仅分割出了目标，而且对这些目标像素非常笃定（概率极高）。
                如果返回值为 0.55：说明模型虽然勉强把这些像素划为了前景（刚刚超过 0.5），但内心非常犹豫，置信度很低。
                返回值接近 0：说明画面中几乎没有检测到任何前景目标。
            ''' 
            if self.use_cca:
                _pred = np.array(pred.detach().cpu())
                _pred, conf = cca(_pred, output_logits, return_conf=True)
                pred = torch.from_numpy(_pred)
            if self.training:
                return output_logits, [conf]
            return pred, [conf]

        _sam3_pil = None
        if getattr(self, "_sam3_backend", False):
            from models.sam3_grounding import query_chw_tensor_to_pil

            _sam3_pil = query_chw_tensor_to_pil(query_image[0])

        # 如果查询图像尺寸不等于模型设定的 `self.image_size`，
        # 就把图像和粗分割 logits 值到统一尺寸(保证后续 SAM 处理时尺寸一致）
        # 通常固定需要例如 1024*1024 或 256*256 的尺寸，由 self.image_size 定义
        if query_image.shape[-2:] != self.image_size:
            query_image = F.interpolate(query_image, size=self.image_size, mode='bilinear')
            output_logits = F.interpolate(output_logits, size=self.image_size, mode='bilinear')
        # if need_softmax(output_logits):
        # output_logits = output_logits.softmax(dim=1)

        #- 然后对 logits 做 softmax 得到概率图 `output_p`
        #- 再 argmax 得到离散预测 `pred`
        # output_p = output_logits
        output_p = output_logits.softmax(dim=1)
        pred = output_p.argmax(dim=1)[0]

        # # ABLATION_V2_PATCHED
        # === ABLATION: oracle_coarse ===
        #__含义__：不用粗分割模型（ALPNet）的预测，而是直接用真实标签（gt_mask）当作"完美的粗分割结果"。
        #粗分割模型的上限/瓶颈
        if self.ablation_mode == "oracle_coarse" and gt_mask is not None:
            _gt_r = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0).float(),
                                  size=output_logits.shape[-2:], mode='nearest')[0][0]
            output_logits = torch.zeros_like(output_logits)
            output_logits[0, 1] = _gt_r * 20
            output_logits[0, 0] = (1 - _gt_r) * 20
            output_p = output_logits.softmax(dim=1)
            pred = output_p.argmax(dim=1)[0]
        # === ABLATION: oracle_prompt (early return) ===
        #不用模型自己从粗分割里提取提示词（点/框）
        #从真值掩码算出"完美"的提示词——真值目标的包围盒和中心点，喂给 SAM。
        if self.ablation_mode == "oracle_prompt" and gt_mask is not None:
            _gt_np = gt_mask.cpu().numpy().astype(np.uint8)
            if _gt_np.max() == 0:
                _ep = F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(), size=original_size, mode='nearest')[0][0]
                return _ep, [0]
            _gt_r = cv2.resize(_gt_np, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            _ys, _xs = np.where(_gt_r > 0)
            _bbox = np.array([_xs.min(), _ys.min(), _xs.max(), _ys.max()])
            _bboxes = np.array([_bbox])
            _cy, _cx = int(_ys.mean()), int(_xs.mean())
            _sam_pts = np.array([[[_cx, _cy]]])
            _sam_neg_pts = [None]
            _qry = query_image
            if self.sam_trans is not None:
                _qry = self.sam_trans.apply_image_torch(_qry[0])
                _qry = self.sam_trans.preprocess(_qry)
                _qry = _qry.permute(1, 2, 0).detach().cpu().numpy()
            else:
                _qry = _qry.permute(1, 2, 0).detach().cpu().numpy()
            _qry = ((_qry - _qry.min()) / (_qry.max() - _qry.min()) * 255).astype(np.uint8)
            if getattr(self, "_sam3_backend", False):
                from models.sam3_grounding import predict_grounding_mask, query_chw_tensor_to_pil

                _pil = _sam3_pil if _sam3_pil is not None else query_chw_tensor_to_pil(query_image[0])
                _gt_o = gt_mask.cpu().numpy().astype(np.uint8)
                _ys, _xs = np.where(_gt_o > 0)
                _box_o = [int(_xs.min()), int(_ys.min()), int(_xs.max()), int(_ys.max())]
                _mask, _sc = predict_grounding_mask(self.sam3_runtime, _pil, _box_o)
                if self.training:
                    _masks = [np.where(_mask, 1.0, -1.0).astype(np.float32)]
                else:
                    _masks = [_mask.astype(np.float32)]
                _scores = [_sc]
            else:
                _masks, _scores = self.predict_w_points_bbox(
                    _sam_pts, _bboxes, _sam_neg_pts, _qry, pred,
                    return_logits=True if self.training else False,
                )
            _pred_out = sum(_masks)
            if not self.training:
                _pred_out = _pred_out > 0
            _pred_out = torch.tensor(_pred_out).float().to(output_p.device)
            _pred_out = F.interpolate(_pred_out.unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest')[0][0]
            return _pred_out, _scores
        # === ABLATION: oracle_mask (early return) ===
        #不用模型生成的粗分割掩码，把真值掩码当作 SAM 的 mask 提示。
        if self.ablation_mode == "oracle_mask" and gt_mask is not None:
            _gt_np = gt_mask.cpu().numpy().astype(np.uint8)
            if _gt_np.max() == 0:
                _ep = F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(), size=original_size, mode='nearest')[0][0]
                return _ep, [0]
            _gt_r = cv2.resize(_gt_np, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            _sam_masks = np.stack([_gt_r.astype(np.float32)])
            _qry = query_image
            if self.sam_trans is not None:
                _qry = self.sam_trans.apply_image_torch(_qry[0])
                _qry = self.sam_trans.preprocess(_qry)
                _qry = _qry.permute(1, 2, 0).detach().cpu().numpy()
            else:
                _qry = _qry.permute(1, 2, 0).detach().cpu().numpy()
            _qry = ((_qry - _qry.min()) / (_qry.max() - _qry.min()) * 255).astype(np.uint8)
            _masks, _scores = self.predict_w_masks(_sam_masks, _qry, original_size)
            _pred_out = sum(_masks)
            if not self.training:
                _pred_out = _pred_out > 0
            _pred_out = torch.tensor(_pred_out).float().to(output_p.device)
            _pred_out = F.interpolate(_pred_out.unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest')[0][0]
            return _pred_out, _scores
        # === ABLATION: soft_mask (early return) ===
        #取粗分割的前景概率 `output_p[0,1]` --> 缩放到 256×256（SAM 的 mask 输入尺寸） --> 直接喂给 SAM 的 `mask_input`
        if self.ablation_mode == "soft_mask":
            _fg_prob = output_p[0, 1].detach().cpu().numpy()
            if _fg_prob.max() < 0.01:
                _ep = F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(), size=original_size, mode='nearest')[0][0]
                return _ep, [0]
            _mask_input = cv2.resize(_fg_prob, (256, 256), interpolation=cv2.INTER_LINEAR)
            _mask_input = ((_mask_input - 0.5) * 20).astype(np.float32)
            _mask_input = _mask_input[None, ...]
            _qry = query_image
            if self.sam_trans is not None:
                _qry = self.sam_trans.apply_image_torch(_qry[0])
                _qry = self.sam_trans.preprocess(_qry)
                _qry = _qry.permute(1, 2, 0).detach().cpu().numpy()
            else:
                _qry = _qry.permute(1, 2, 0).detach().cpu().numpy()
            _qry = ((_qry - _qry.min()) / (_qry.max() - _qry.min() + 1e-8) * 255).astype(np.uint8)
            self.predictor.set_image(_qry)
            _masks, _scores, _ = self.predictor.predict(
                mask_input=_mask_input,
                multimask_output=True
            )
            _best_idx = int(np.argmax(_scores))
            _pred_out = _masks[_best_idx]
            if not self.training:
                _pred_out = _pred_out > 0
            _pred_out = torch.tensor(_pred_out).float().to(output_p.device)
            _pred_out = F.interpolate(_pred_out.unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest')[0][0]
            return _pred_out, [float(_scores[_best_idx])]
        # === ABLATION: soft_prompt (flag, no early return) ===
        #验证"软掩码 + 点/框"组合提示的效果，而不是单独用软掩码。
        _soft_mask_input = None
        if self.ablation_mode.startswith("soft_prompt"):
            _fg_prob = output_p[0, 1].detach().cpu().numpy()
            _soft_mask_input = build_soft_prompt_mask_input(_fg_prob, self.ablation_mode)
        # === END ABLATION BLOCK ===
        '''
        消融模式	替换/修改的环节	是否提前返回
        oracle_coarse	粗分割 → 真值掩码	否（继续走 SAM） 【把正确答案当成草图发下去。我们来看看，如果草图是100%正确的，后续提点、SAM 抠图这一套流程能不能完美运作】
        oracle_prompt	提示词 → 真值包围盒+中心点	是 【直接根据真实的 GT 答案，算出绝对精准的中心点和边界框 (Bounding Box)，把这俩坐标直接塞给 SAM。测试提示词算法】
        oracle_mask	掩码提示 → 真值掩码	是 【提取真实的 GT 答案并缩放到 256 * 256（SAM 的专属低分率掩码输入格式），作为底图直接塞给 SAM（不给点和框）】
        soft_mask	掩码提示 → 软概率图	是 【原始的概率热力图（比如 0.8、0.4、0.2），直接作为 mask_input 喂给 SAM。对比软概率信息（Soft）与硬掩码（Hard 0/1）】
        soft_prompt	掩码提示 → 软概率图（组合）	否 【验证“几何坐标（锁定位置） + 软概率图（提供形状模糊预期）”多模态提示】
        这些模式共同回答一个问题："如果某个环节是完美的，最终效果能到多少？" 通过对比各 oracle 模式与正常流程的差距，就能定位系统的性能瓶颈在哪一环。
        '''

        # 步骤 5：连通域分析 (CCA)
        # 寻找粗分割二值预测图中的所有连通区域。这一步非常关键，它将“一团团”的像素归类为一个个独立的实例/目标，方便后续针对每个目标分别生成提示。
        _pred = np.array(output_p.argmax(dim=1)[0].detach().cpu()) 
        start_time = time.time()
        if self.use_cca: # 只保留置信度最高的那唯一一个岛屿，放弃其他所有东西。送给 SAM 的提示词只有一组。 
            _min_area = 500 if 'area_filter' in self.ablation_fixes else 0
            conn_components = cca(_pred, output_logits, return_cc=True, min_area=_min_area)
            conf=None
        else: #策略 A：多目标保留模式（use_cca = False）
            conn_components, conf = get_connected_components(_pred, output_logits, return_conf=True)
        if self.debug:
            plot_connected_components(conn_components, query_image[0,0].detach().cpu(), conf)
        # print(f"connected components took {time.time() - start_time} seconds")
        if _pred.max() == 0: # 如果粗分割完全没有预测出前景，直接返回空掩码
            _empty_pred = output_p.argmax(dim=1)[0]
            _empty_pred = F.interpolate(_empty_pred.unsqueeze(0).unsqueeze(0).float(), size=original_size, mode='nearest')[0][0]
            return _empty_pred, [0]
        
        # 步骤 6：生成几何提示词（边界框）
        # 遍历连通域，如果启用了 use_bbox，则为每个连通区域计算出准确的边界框坐标。
        if self.use_bbox:
            start_time = time.time()
            try:
                bboxes = self.get_bbox_per_cc(conn_components) 
            except:
                bboxes = [None] * conn_components[0]
        else:
            bboxes = [None] * conn_components[0]
        # print(f"getting bboxes took {time.time() - start_time} seconds")

        # 步骤 7：生成几何提示词（点和掩码）
        # 遍历连通域，如果启用了 use_points，提取高置信度前景点（和负样本背景点）；如果启用了 use_mask，则提取整个连通域的形状作为低分辨率掩码。
        start_time = time.time()
        if self.use_points:
            sam_input_points, sam_input_point_labels, sam_neg_input_points, sam_neg_input_labels = self.get_sam_input_points(conn_components, output_p, get_neg_points=self.use_neg_points, l=1)
            if self.ablation_mode.endswith("_mnn"):
                sam_input_points = self.replace_confidence_point_with_mnn(
                    sam_input_points, conn_components
                )
        else:
            sam_input_points = [None] * conn_components[0]
            sam_input_point_labels = [None] * conn_components[0]
            sam_neg_input_points = [None] * conn_components[0]
            sam_neg_input_labels = [None] * conn_components[0]
        # print(f"getting sam input points took {time.time() - start_time} seconds")
        
        if self.use_mask:
            sam_input_masks, sam_input_mask_labels = self.get_sam_input_mask(conn_components) 
        else:
            sam_input_masks = None
            sam_input_mask_labels = None
            
        if self.debug and sam_input_points is not None:
            title = f'debug/most_conf_points.png'
            if self.use_cca:
                title = f'debug/most_conf_points_cca.png'
            # convert points to a list where each item is a list of 2 elements in xy format
            self.plot_most_conf_points(sam_input_points, None, _pred, query_image[0, 0].detach().cpu(), bboxes=bboxes, title=title) # TODO add plots for all points not just the first set of points

        # 步骤 8–9：SAM / SAM3 精修
        if getattr(self, "_sam3_backend", False):
            if self.use_mask:
                raise NotImplementedError(
                    "SAM3 grounding backend supports text+bbox (T+I) only; disable use_mask or use SAM1."
                )
            masks, scores = [], []
            if self.use_points or self.use_bbox:
                if not self.use_bbox:
                    raise NotImplementedError(
                        "SAM3 grounding requires coarse bounding boxes (use_bbox=True)."
                    )
                masks, scores = self.predict_w_bbox_sam3(
                    bboxes,
                    _sam3_pil,
                    self.image_size,
                    original_size,
                    sam_input_points=sam_input_points,
                    return_logits=bool(self.training),
                )
        else:
            # SAM1：图像预处理为 uint8，再送入 SamPredictor
            if self.sam_trans is None:
                query_image = query_image.permute(1, 2, 0).detach().cpu().numpy()
            else:
                query_image = self.sam_trans.apply_image_torch(query_image[0])
                query_image = self.sam_trans.preprocess(query_image)
                query_image = query_image.permute(1, 2, 0).detach().cpu().numpy()

            query_image = (
                (query_image - query_image.min())
                / (query_image.max() - query_image.min() + 1e-8)
                * 255
            ).astype(np.uint8)

            if self.use_mask:
                masks, scores = self.predict_w_masks(sam_input_masks, query_image, original_size)

            start_time = time.time()
            if self.use_points or self.use_bbox:
                masks, scores = self.predict_w_points_bbox(
                    sam_input_points,
                    bboxes,
                    sam_neg_input_points,
                    query_image,
                    pred,
                    return_logits=True if self.training else False,
                    sam_mask_input=_soft_mask_input,
                )
        # print(f"predicting w points/bbox took {time.time() - start_time} seconds")

        # 步骤 10：结果融合与输出尺寸还原
        # 将 SAM 对不同目标输出的多个 Mask 累加合并成一张图，进行阈值化（非训练模式下），最后通过插值算法恢复到输入时的初始分辨率，输出最终的高清分割结果。    
        if 'weighted_agg' in self.ablation_fixes:
            pred = sum(m * s for m, s in zip(masks, scores))
        else:
            pred = sum(masks)
        if not self.training:
            pred = pred > 0
        pred = torch.tensor(pred).float().to(output_p.device)
        
        # pred = torch.tensor(masks[0]).float().cuda()
        # resize pred to the size of the input
        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest')[0][0]
        
        return pred, scores
'''
SamPredictor 不是单纯的“解码器（Decoder）”，它其实是一个封装了整个 SAM 模型全链路的官方推理调度器（Wrapper/Interface）。
从 SAM（Segment Anything Model）的底层第一性原理来看，完整的 SAM 模型包含三个核心神经网络组件：
图像编码器（Image Encoder）：体积庞大、计算极其耗时的 ViT 结构，负责把整张图片压缩成高维特征图（Image Embedding）。
提示编码器（Prompt Encoder）：负责把传入的点、框或文本，转化为统一的特征向量。
掩码解码器（Mask Decoder）：一个非常轻量的网络，负责把上面两步产生的“图像特征”和“提示特征”进行融合，最终“解出”分割掩码（Mask）。

在提供的代码中，SamPredictor 作为“总管”，是将这三个部分串联起来供开发者使用的黑盒。
你可以通过它暴露的两个核心 API 来看清它的分工：调用图像编码器：当代码执行 self.predictor.set_image(qry_img) 时，
SamPredictor 在底层调用的是图像编码器。这一步会进行最重的计算，算出图像特征并缓存在内存里，这样不管后面给多少个不同的提示点，这步沉重的计算都只需要做一次。  
调用提示编码器 + 解码器：当代码执行 self.predictor.predict(point_coords=...) 时，SamPredictor 在底层实际上是先调用了提示编码器将你传入的提示点转化，
然后再调用了掩码解码器（Mask Decoder）输出最终的预测结果。
'''       
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def need_softmax(tensor, dim=1):
    return not torch.all(torch.isclose(tensor.sum(dim=dim), torch.ones_like(tensor.sum(dim=dim))) & (tensor >= 0))

        
        
        
        
