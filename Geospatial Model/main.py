from functools import partial
from mmcv.runner import load_checkpoint
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Block
from typing import List
import numbers
from math import cos, pi
from typing import Callable, List, Optional, Union
import torch
import torch.nn as nn
import yaml
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple

import numpy as np

from einops import rearrange

import torch
import torch.nn as nn
from einops import rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F

from collections import OrderedDict
#import mmcv
import numpy as np
import torch
import random

# Set the fixed seed
seed = 123

# NumPy
np.random.seed(seed)

# Python Random
random.seed(seed)

# PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        #label = torch.from_numpy(
         #   mmcv.imread(label, flag='unchanged', backend='pillow'))
         label = torch.from_numpy(label)
    else:
        label = torch.from_numpy(label)

    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
    if label_map is not None:
        label_copy = label.clone()
        for old_id, new_id in label_map.items():
            label[label_copy == old_id] = new_id
    
    
    # Create a mask where label is not equal to 2
    mask = (label != 2)
    
    # Apply the mask to filter pred_label and label
    pred_label = pred_label[mask]
    label = label[mask]
    
    
    #mask = (label != ignore_index)
    
    #center_row = pred_label.shape[0] // 2
    #center_col = pred_label.shape[1] // 2
    #pred_label = pred_label[center_row, center_col]
    #label = label[center_row, center_col]
    #pred_label = pred_label[mask]
    #label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean F-Score (mFscore)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics
    
    
    
    
    
    
    
    
    

class Residual3DConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size, stride, padding),
            nn.BatchNorm3d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.residual_conv = nn.Conv3d(in_chans, out_chans, 1) if in_chans != out_chans else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual_conv(x)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: 3d tuple of grid size: t, h, w
    return:
    pos_embed: L, D
    """

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _convTranspose2dOutput(
    input_size: int,
    stride: int,
    padding: int,
    dilation: int,
    kernel_size: int,
    output_padding: int,
):
    """
    Calculate the output size of a ConvTranspose2d.
    Taken from: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    """
    return (
        (input_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )

class PatchEmbed(nn.Module):
    """ Frames of 2D Images to Patch Embedding
    The 3D version of timm.models.vision_transformer.PatchEmbed
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            num_frames=1,
            tubelet_size=1,
            in_chans=43,
            embed_dim=1024,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = (num_frames // tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
                              stride=(tubelet_size, patch_size[0], patch_size[1]), bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x




class TemporalViTEncoder(nn.Module):
    """Encoder from an ViT with capability to take in temporal input.

    This class defines an encoder taken from a ViT architecture.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 1,
        in_chans: int = 45,
        embed_dim: int = 768,
        depth: int = 24,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_pix_loss: bool = False,
        pretrained: str = None
    ):
        """

        Args:
            img_size (int, optional): Input image size. Defaults to 224.
            patch_size (int, optional): Patch size to be used by the transformer. Defaults to 16.
            num_frames (int, optional): Number of frames (temporal dimension) to be input to the encoder. Defaults to 1.
            tubelet_size (int, optional): Tubelet size used in patch embedding. Defaults to 1.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Embedding dimension. Defaults to 1024.
            depth (int, optional): Encoder depth. Defaults to 24.
            num_heads (int, optional): Number of heads used in the encoder blocks. Defaults to 16.
            mlp_ratio (float, optional): Ratio to be used for the size of the MLP in encoder blocks. Defaults to 4.0.
            norm_layer (nn.Module, optional): Norm layer to be used. Defaults to nn.LayerNorm.
            norm_pix_loss (bool, optional): Whether to use Norm Pix Loss. Defaults to False.
            pretrained (str, optional): Path to pretrained encoder weights. Defaults to None.
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.num_frames = num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
       # self.multi_head_self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm_pix_loss = norm_pix_loss
        self.pretrained = pretrained

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if isinstance(self.pretrained, str):
            self.apply(self._init_weights)
            print(f"load from {self.pretrained}")
            load_checkpoint(self, self.pretrained, strict=False, map_location="cpu")
            
        elif self.pretrained is None:
            # # initialize nn.Linear and nn.LayerNorm
            self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed patches
        
       
        x = self.patch_embed(x)
        

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return tuple([x])
        
        
class ConvTransformerTokensToEmbeddingNeck(nn.Module):
    """
    Neck that transforms the token-based output of transformer into a single embedding suitable for processing with standard layers.
    Performs 4 ConvTranspose2d operations on the rearranged input with kernel_size=2 and stride=2
    """

    def __init__(
        self,
        embed_dim: int,
        output_embed_dim: int,
        # num_frames: int = 1,
        Hp: int = 14,
        Wp: int = 14,
        drop_cls_token: bool = True,
    ):
        """

        Args:
            embed_dim (int): Input embedding dimension
            output_embed_dim (int): Output embedding dimension
            Hp (int, optional): Height (in patches) of embedding to be upscaled. Defaults to 14.
            Wp (int, optional): Width (in patches) of embedding to be upscaled. Defaults to 14.
            drop_cls_token (bool, optional): Whether there is a cls_token, which should be dropped. This assumes the cls token is the first token. Defaults to True.
        """
        super().__init__()
        self.drop_cls_token = drop_cls_token
        self.Hp = Hp
        self.Wp = Wp
        self.H_out = Hp
        self.W_out = Wp
        # self.num_frames = num_frames

        kernel_size = 2
        stride = 2
        dilation = 1
        padding = 0
        output_padding = 0
        for _ in range(4):
            self.H_out = _convTranspose2dOutput(
                self.H_out, stride, padding, dilation, kernel_size, output_padding
            )
            self.W_out = _convTranspose2dOutput(
                self.W_out, stride, padding, dilation, kernel_size, output_padding
            )

        self.embed_dim = embed_dim
        self.output_embed_dim = output_embed_dim
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )

    def forward(self, x):
        
        x = x[0]
        #print(x.shape)
        if self.drop_cls_token:
            x = x[:, 1:, :]
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1 , self.Hp, self.Wp)

        x = self.fpn1(x)
        x = self.fpn2(x)

        x = x.reshape((-1, self.output_embed_dim, self.H_out, self.W_out))

        out = tuple([x])

        return out
        
        
class Norm2d(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
        



import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels
        self.norm_cfg=dict(type="BN", requires_grad=True)
        conv_padding = (kernel_size // 2) * dilation
        convs = []
        for i in range(num_convs):
            _in_channels = self.in_channels if i == 0 else self.channels
            convs.append(
                ConvModule(
                    _in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output




class ImageToTextCrossAttention(nn.Module):
    def __init__(self, image_dim=768, text_dim=768, hidden_dim=512):
        super().__init__()
        self.query_proj = nn.Linear(image_dim, hidden_dim)
        self.key_proj = nn.Linear(text_dim, hidden_dim)
        self.value_proj = nn.Linear(text_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, image_dim)

    def forward(self, image_features, text_embeddings):
        """
        image_features: [B, C, H, W]
        text_embeddings: [T, text_dim] — shared across batch
        """
        B, C, H, W = image_features.shape
        N = H * W
        T = text_embeddings.shape[0]

        x_flat = image_features.flatten(2).transpose(1, 2)   # [B, N, C]
        Q = self.query_proj(x_flat)                          # [B, N, hidden]

        # Expand text embeddings to match batch
        text_expanded = text_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, T, text_dim]
        K = self.key_proj(text_expanded)                                # [B, T, hidden]
        V = self.value_proj(text_expanded)                              # [B, T, hidden]

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)  # [B, N, T]
        attn_weights = F.softmax(attn_scores, dim=-1)                        # [B, N, T]

        attended_text = torch.bmm(attn_weights, V)  # [B, N, hidden]
        out = self.out_proj(attended_text)          # [B, N, image_dim]
        out = out.transpose(1, 2).view(B, C, H, W)   # [B, C, H, W]
        
        return out  # Injected text-guided enhancement


class TemporalViTModel(nn.Module):
    def __init__(self, num_classes, img_size=256, patch_size=16, num_frames=1, embed_dim=768, output_embed_dim=768, pretrained='/nfs/turbo/coe-ecbk/jowaria/masked_autoencoder_flowallarc.pth'):
        super().__init__()
        # Encoder
        self.encoder = TemporalViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            embed_dim=embed_dim,
            pretrained=pretrained
        )
        # Neck
        self.neck = ConvTransformerTokensToEmbeddingNeck(
            embed_dim=embed_dim,
            output_embed_dim=output_embed_dim,
            drop_cls_token=True,
            Hp=16,
            Wp=16
        )
        # Head
        self.head = FCNHead(
           in_channels=output_embed_dim,
            channels=256,
            num_classes=num_classes,
            num_convs = 1,
            in_index = -1,
            concat_input=False,
        dropout_ratio=0.1,
        align_corners=False
        )
        
        self.head2 = FCNHead(
            in_channels=output_embed_dim,
            channels=256,
            num_classes=num_classes,
            num_convs = 2,
            in_index = -1,
            concat_input=False,
        dropout_ratio=0.1,
        align_corners=False
        )
        
        self.cross_attn = ImageToTextCrossAttention(
    image_dim=output_embed_dim,
    text_dim=768,
    hidden_dim=512
)


    def forward(self, x, text_embeddings):
        x = self.encoder(x)  # Output is a tuple
        x = self.neck(x)     # Output is a tuple
        
        if isinstance(x, tuple):
            fused_feat = x[0] + self.cross_attn(x[0], text_embeddings)
            x = (fused_feat,) + x[1:]  # reconstruct the tuple with modified first element
        else:
            x = x + self.cross_attn(x, text_embeddings)

        # Step 3: Fuse (e.g., additive or concat+conv)
        #x = x + text_guided  # residual fusion
        x1 = self.head(x)  # Process first element of tuple
        x2 = self.head2(x)
        return x1,x2



import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import glob
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import yaml
from tqdm import tqdm





class CustomLoss(nn.Module):
    def __init__(self, loss_name='loss_custom', ignore_index = None, class_weights=None):
        super(CustomLoss, self).__init__()
        self._loss_name = loss_name
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

    def generate_noise_mask_path(self, image_path):
        """
        Modify the image file path to point to the corresponding noise mask.

        Args:
            image_path (str): Original image file path.

        Returns:
            str: Corresponding noise mask file path.
        """
        # Replace 'images' with 'noise_masks' and '_merged.tif' with '_masks.tif'
        noise_mask_path = image_path.replace('/images_flowdir_arc/', '/noise_masks_dirdownaa_diff_whole/').replace('_merged.tif', '_mask.tif')
        return noise_mask_path

    def load_noise_mask(self, file_path):
        """
        Load a noise mask from a TIFF file using rasterio and convert it into a tensor.

        Args:
            file_path (str): Path to the noise mask TIFF file.

        Returns:
            torch.Tensor: Noise mask tensor of shape (H, W).
        """
        with rasterio.open(file_path) as src:
            noise_mask_array = src.read(1)  # Reading the first band (assuming single-band noise mask)
            noise_mask_tensor = torch.from_numpy(noise_mask_array).float()  # Convert to PyTorch tensor
        return noise_mask_tensor
        
        

    def forward(self, predictions, label_mask, image_paths, weight=None, avg_factor=None, reduction_override=None, **kwargs): #where image_paths are the paths to the noise masks associated to the label masks
        """
        Forward method to compute the custom loss with uncertainty weighting.

        Args:
            predictions (torch.Tensor): Model predictions (logits), shape (batch_size, num_classes, H, W).
            label_mask (torch.Tensor): Ground truth mask, shape (batch_size, H, W).
            image_paths (list of str): List of file paths to image files, one per image in the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        
        device = predictions.device
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)
            
        batch_size = predictions.shape[0]
        noise_masks = []

        # Load each noise mask from the modified image file paths
        for i in range(batch_size):
            # Generate the noise mask path from the image path
            noise_mask_path = self.generate_noise_mask_path(image_paths[i])
            
            # Load the noise mask from the generated file path
            noise_mask = self.load_noise_mask(noise_mask_path)
            
            noise_masks.append(noise_mask)

        # Stack noise masks to form a batch (batch_size, H, W)
        noise_masks = torch.stack(noise_masks).to(predictions.device)  # Ensure the noise masks are on the same device
        
            
        BCE_loss = nn.CrossEntropyLoss(weight=self.class_weights)(predictions, label_mask)
        pt = torch.exp(-BCE_loss)  # Prevents underflow by directly computing pt
        F_loss = 1 * (1 - pt) ** 2 * BCE_loss
        
        
        # Weight the loss by the noise masks
        weighted_loss = (F_loss)*noise_masks # Apply noise masks as weights

        # Compute the mean loss
        loss = torch.mean(weighted_loss)
        
        # Calculate ECE
#        ece_loss = differentiable_ece1(predictions, label_mask)
    
        # Combine Cross Entropy Loss and ECE (can add weighting factors if needed)
        total_loss = loss # Adjust with coefficients if needed, e.g., 0.9 * loss + 0.1 * ece_loss
    
        return total_loss
        


# Example class weights (adjust as needed)
class_weights = [1.0, 0.07, 0.03]  # Modify based on your dataset's class distribution

# Instantiate the custom loss function
criterion = CustomLoss(ignore_index=-1, class_weights=class_weights)

# In your training or inference script
data = torch.load('roberta_pfas_embeddings_d_l_5.pt')  # shape: [602, 768]
text_embeddings = data["mean_embeddings"] 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text_embeddings = text_embeddings.float().to(device)


def preprocess_image(image, means, stds):
    means = np.array(means).reshape(-1, 1, 1)
    stds = np.array(stds).reshape(-1, 1, 1)
    # normalize image
    normalized = image.copy()
    normalized = ((image - means) / stds)
    normalized = torch.from_numpy(normalized.reshape(normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
    return normalized



class MultibandTiffDataset(Dataset):
    def __init__(self, image_directories, label_directories, bands, data_mean, data_std, random_cropping=False):
        """
        Multiband TIFF Dataset with image paths, labels, and optional random cropping.

        Args:
            image_directories (list of str): Directories containing image files.
            label_directories (list of str): Directories containing corresponding label masks.
            bands (list of int): List of bands to extract from each image.
            data_mean (list of float): Mean values for normalization (one per band).
            data_std (list of float): Std dev values for normalization (one per band).
            random_cropping (bool): Whether to apply random cropping.
        """
        self.image_paths = []
        self.label_paths = []

        # Collect image and label file paths
        for image_dir, label_dir in zip(image_directories, label_directories):
            image_files = glob.glob(os.path.join(image_dir, '*.tif'))
            for image_file in image_files:
                # Derive the corresponding label path
                label_file = os.path.join(label_dir, os.path.basename(image_file).replace('_merged.tif', '_mask.tif'))
                if os.path.exists(label_file):  # Ensure label exists for the image
                    self.image_paths.append(image_file)
                    self.label_paths.append(label_file)

        self.bands = bands
        self.data_mean = torch.tensor(data_mean, dtype=torch.float32)
        self.data_std = torch.tensor(data_std, dtype=torch.float32)
        self.random_cropping = random_cropping
        if self.random_cropping:
            self.random_crop = transforms.RandomCrop(224)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load image
        with rasterio.open(image_path) as src:
            img = np.stack([src.read(band) for band in self.bands], axis=0)  # Shape: (num_bands, H, W)

        # Load label mask
        with rasterio.open(label_path) as src:
            label = src.read(1)  # Assuming the label mask is a single-band TIFF (H, W)

        # Preprocess image
        img = preprocess_image(img, self.data_mean, self.data_std)
        
        #img = tuple(preprocess_image(band, self.data_mean, self.data_std) for band in img)

        # Random cropping (applied to both image and label)
        if self.random_cropping:
            crop = self.random_crop(torch.tensor(img))  # Apply random crop to image
            img, label = crop, self.random_crop(torch.tensor(label))

        # Return image, label, and image path
        return {
            'images': img,
            #'images': img,
            'labels': torch.tensor(label, dtype=torch.long),
            'image_paths': image_path
        }



train_image_dirs = ["/nfs/turbo/coe-ecbk/jowaria/train_combined_nsra/train_latest_water_elev_2022_new_new_filtered/train_latest_new_new_256n/images_flowdir_arc"]
train_label_dirs = ["/nfs/turbo/coe-ecbk/jowaria/train_combined_nsra/train_latest_water_elev_2022_new_new_filtered/train_latest_new_new_256n/masks"]
test_image_dirs = ["/nfs/turbo/coe-ecbk/jowaria/train_combined_nsra/train_latest_water_elev_2022_new_new_filtered/test_latest_new_new_256n/images_flowdir_arc"]
test_label_dirs = ["/nfs/turbo/coe-ecbk/jowaria/train_combined_nsra/train_latest_water_elev_2022_new_new_filtered/test_latest_new_new_256n/updated_masks_sparse"]



data_mean = [
    54.43250633548725, 2420.615977724663, 4752.945829880318, 3797.5476023258507, 1647.7756037563736,
    7733.128014998057, 1252.4540875382959, 3767.8312476991146, 1993.0822415062196, 1152.605698842765,
    4126.689650391868, 5786.144238103053, 2056.401946591497, 4078.908179978712, 3335.7994021693653,
    1392.0515087190117, 2342.4641368437287, 1991.8751616636737, 287.177976571065, 2043.6217294338817,
    1843.2161662753272, 2064.0993287855704, 721.5768764855314, 1754.3789413166392, 838.1923445591713,
    4096.967526185525, 2229.661188808358, 743.2914138475984, 8428.159249173666, 2618.7688700063427,
    1252.0052939098755, 3881.1992129946243, 4047.0715717093494, 6088.370196704393, 7608.615934962294,
    7098.36192131094, 9050.875210118891, 4844.057112814508, 7741.941483484172, 8638.421529584777,
    8802.556607971823, 32404.759609765642, 39897.91062280858, 30.414002843363633, 2.3468280447606356
]
#0.3677128111461984
data_std = [
    24.901177948348295, 2476.5827093221005, 4669.099437578338, 2738.439051010196, 1501.7596676088674,
    8004.304020677584, 1343.4646401983364, 3776.9322807366075, 2198.5583937961264, 1305.6678364455927,
    3880.062457093438, 4554.855174501481, 2075.9400631678036, 3505.988183672428, 3615.460317277773,
    1208.1776005947338, 2917.3776738153992, 2198.0235843500577, 215.45968979679793, 1611.406426698672,
    1198.815692346246, 2237.415367957372, 935.9017162027332, 1507.6917784786533, 802.9713439246704,
    3484.90940022616, 1883.7836435215015, 836.2647023104796, 5266.686368507145, 2589.9777496997103,
    1231.7731123982749, 3523.4810936534736, 2803.9851173371803, 4332.076762567314, 6666.086580897644,
    6904.308927031156, 8792.432774999806, 3837.5486447656244, 5930.885273016913, 6930.863107733796,
    5729.460991789289, 19323.50666403787, 18972.940960531276, 43.70969837136813, 0.4759604522732434
]
#0.482182641397584


bands = [1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41, 42,43,44,45]

# Training dataset
train_dataset = MultibandTiffDataset(
    image_directories=train_image_dirs,
    label_directories=train_label_dirs,
    bands=bands,
    data_mean=data_mean,
    data_std=data_std,
    random_cropping= False  # Enable random cropping for training
)

# Testing dataset
test_dataset = MultibandTiffDataset(
    image_directories=test_image_dirs,
    label_directories=test_label_dirs,
    bands=bands,
    data_mean=data_mean,
    data_std=data_std,
    random_cropping=False  # No cropping for testing
)


# DataLoader parameters
batch_size = 4
num_workers = 7

# Training DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

# Testing DataLoader
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import os


def evaluate_segmentation(predictions, labels, num_classes=3, ignore_index=-1, beta=1):
    """
    Evaluates segmentation predictions using IoU, F1 score, accuracy, precision, and recall.
    
    Args:
        predictions: List of predicted segmentation maps or tensors.
        labels: List of ground truth segmentation maps or tensors.
        num_classes: Number of classes (default is 3).
        ignore_index: Index to ignore in evaluation (e.g., background class).
        beta: Beta value for F-score calculation (default 1 for F1 score).
    
    Returns:
        Dictionary containing the evaluation metrics: IoU, F1 score, accuracy, precision, and recall.
    """
    # Convert predictions and labels into a suitable format
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Calculate mean IoU
    iou_result = mean_iou(
        results=predictions,
        gt_seg_maps=labels,
        num_classes=num_classes,
        ignore_index=ignore_index
    )

    # Calculate mean F-score
    fscore_result = mean_fscore(
        results=predictions,
        gt_seg_maps=labels,
        num_classes=num_classes,
        ignore_index=ignore_index,
        beta=beta
    )

    # Assemble the results into a dictionary
    eval_metrics = {
        'iou': iou_result['IoU'],  # Mean IoU for each class
        'fscore': fscore_result['Fscore'],  # F1 score for each class
        'precision': fscore_result['Precision'],  # Precision (macro average)
        'recall': fscore_result['Recall'],  # Recall (macro average)
        'aAcc': fscore_result['aAcc']
    }

    return eval_metrics


import numbers
from math import cos, pi
from typing import Callable, List, Optional, Union

import mmcv
from mmcv import runner
#from .hook import HOOKS, Hook


class LrUpdaterHook():
    """LR Scheduler in MMCV.

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(self,
                 by_epoch: bool = True,
                 warmup: Optional[str] = None,
                 warmup_iters: int = 6,
                 warmup_ratio: float = 1e-06,
                 warmup_by_epoch: bool = True) -> None:
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant", "linear" and "exp"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters: Optional[int] = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs: Optional[int] = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr: Union[list, dict] = []  # initial lr for all param groups
        self.regular_lr: list = []  # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(runner.optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        raise NotImplementedError

    def get_regular_lr(self, runner: 'runner.BaseRunner'):
        if isinstance(runner.optimizer, dict):
            lr_groups = {}
            for k in runner.optimizer.keys():
                _lr_group = [
                    self.get_lr(runner, _base_lr)
                    for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters: int):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_run(self, runner: 'runner.BaseRunner'):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                for group in optim.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                _base_lr = [
                    group['initial_lr'] for group in optim.param_groups
                ]
                self.base_lr.update({k: _base_lr})
        else:
            for group in runner.optimizer.param_groups:  # type: ignore
                group.setdefault('initial_lr', group['lr'])
            self.base_lr = [
                group['initial_lr']
                for group in runner.optimizer.param_groups  # type: ignore
            ]

    def before_train_epoch(self, runner: 'runner.BaseRunner'):
        if self.warmup_iters is None:
            epoch_len = len(runner.data_loader)  # type: ignore
            self.warmup_iters = self.warmup_epochs * epoch_len  # type: ignore

        if not self.by_epoch:
            return

        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner: 'runner.BaseRunner'):
        cur_iter = runner.iter
        assert isinstance(self.warmup_iters, int)
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
                
                
class PolyLrUpdaterHook(LrUpdaterHook):

    def __init__(self,
                 power: float = 1.,
                 min_lr: float = 0.,
                 **kwargs) -> None:
        self.power = power
        self.min_lr = min_lr
        super().__init__(**kwargs)

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


class SimpleRunner:
    def __init__(self, optimizer, max_epochs, train_loader):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.data_loader = train_loader
        self.epoch = 0
        self.iter = 0
        self.max_iters = max_epochs * len(train_loader)


lr_scheduler = PolyLrUpdaterHook(
    power=1.0,  # Polynomial decay power
    min_lr=0,  # Minimum learning rate
    by_epoch=True,  # Adjust learning rate per epoch
    warmup='linear',  # Warmup type
    warmup_iters=7,  # Warmup duration (epochs or iterations based on `warmup_by_epoch`)
    warmup_ratio=0.1,  # Starting LR as a fraction of the initial LR
    warmup_by_epoch=True  # Apply warmup per epoch
)


no_data_value = -1
# Define training loop
def train_model2(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, save_dir='./', save_best=True):
    # Initialize best F1 (if needed) for your purposes
    best_f1 = 0
    
    for epoch in range(runner.max_epochs):
        runner.epoch = epoch
        lr_scheduler.before_train_epoch(runner)
  
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        num_batches = 0

        # Training phase
        for batch_idx, data in enumerate(train_loader):
            runner.iter = epoch * len(train_loader) + batch_idx
            lr_scheduler.before_train_iter(runner)
            images = data['images'].to(device)
            labels = data['labels'].to(device)
            image_paths = data['image_paths']

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, aux = model(images,text_embeddings)

            # Compute loss
            total_loss_w = criterion(outputs, labels, image_paths)
            total_loss_w1 = criterion(aux, labels, image_paths)
            loss = total_loss_w + total_loss_w1
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            epoch_losses_with_ece["Total"] += loss.item()
            num_batches += 1

        # Print training loss
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')

        # Save model after every epoch
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, f'check_{epoch+1}.pth'))
        print(f"Model checkpoint saved for epoch {epoch+1}!")

    # Reload and evaluate each checkpoint after training
    for epoch in range(num_epochs):
        # Initialize the model again
        model = TemporalViTModel(num_classes=3).to(device)  # Replace with your actual model class
        model.load_state_dict(torch.load(os.path.join(save_dir, f'check_{epoch+1}.pth')))
        model.eval()  # Set model to evaluation mode
        model = model.to(device)  # Move model to the correct device (CPU or GPU)

        # Evaluate on the test set after reloading each checkpoint
        all_preds = []
        all_labels = []
        test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                images = batch['images'].to(device)
                labels = batch['labels'].to(device)
                image_paths = batch['image_paths']
                outputs, aux = model(images, text_embeddings)
                _, preds = torch.max(outputs, 1)

                all_preds.append(preds)
                all_labels.append(labels)

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Calculate evaluation metrics
        metrics = evaluate_segmentation(all_preds, all_labels)
        print(f"Test Results for Checkpoint at Epoch {epoch+1}:")
        print("Evaluation Metrics:", metrics)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemporalViTModel(num_classes=3).to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
# Scheduler
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-4,
    total_steps=len(train_loader) * 100,
    pct_start=0.4,
    anneal_strategy='cos'
)
runner = SimpleRunner(optimizer=optimizer, max_epochs=100, train_loader=train_loader)

lr_scheduler.before_run(runner)



train_model2(model, train_loader, test_loader, criterion, optimizer, num_epochs=100, save_dir='/nfs/turbo/coe-ecbk/jowaria/train_combined_nsra/models')


