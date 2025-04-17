# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmdet.models.dense_heads import RPNHead
from mmdet.models.utils import multi_apply
from mmdet.structures.bbox import (BaseBoxes, get_box_tensor, get_box_wh,
                                   scale_boxes)
from mmdet.utils import MultiConfig
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbox2hbox
from projects.HERO.hero.hilbert_cross_attention import HilbertCrossScaleAttention
from script.hilber_script_oriented import hilbert_flatten, hilbert_unflatten


@MODELS.register_module()
class HilbertRPNHead(RPNHead):
    """Oriented RPN head for Oriented R-CNN."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 init_cfg: MultiConfig = dict(
                     type='Normal', layer='Conv2d', std=0.01),
                 num_convs: int = 1,
                 use_cross_attention: bool = False,
                 **kwargs) -> None:
        super().__init__(in_channels, num_classes, init_cfg, num_convs, **kwargs)
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.hilbert_cross_attention = HilbertCrossScaleAttention(in_channels)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
            # todo: 这里初始化hilbert卷积
            self.hilbert_conv = nn.Conv1d(self.in_channels, self.feat_channels, 3, padding=1)

        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        reg_dim = self.bbox_coder.encode_size
        self.rpn_reg = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * reg_dim, 1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        # x = self.rpn_conv(x)
        # x = F.relu(x)

        # hilbert重建特征
        x_reconstructed = hilbert_unflatten(x)

        rpn_cls_score = self.rpn_cls(x_reconstructed)
        rpn_bbox_pred = self.rpn_reg(x_reconstructed)
        return rpn_cls_score, rpn_bbox_pred

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        hilbert_seq_list = []
        for n_layer in range(5):
            n, c, h, w = x[n_layer].shape
            x_hilbert = hilbert_flatten(x[n_layer]).view(n, c, -1)
            # todo: 1、这里1*3还是1*9。2、这里用暴力reshape还是unflatten的逆操作
            # x_hilbert = x_hilbert.unfold(dimension=2, size=9, step=1)
            x_hilbert = self.hilbert_conv(x_hilbert)
            hilbert_seq_list.append(x_hilbert)
        if self.use_cross_attention:
            hilbert_seq_list = self.hilbert_cross_attention(hilbert_seq_list)

        return multi_apply(self.forward_single, hilbert_seq_list)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method, which use horizontal bboxes for NMS,
        but return the rotated bboxes result.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert with_nms, '`with_nms` must be True in RPNHead'
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            hbboxes = rbox2hbox(bboxes)
            det_bboxes, keep_idxs = batched_nms(hbboxes, results.scores,
                                                results.level_ids, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
            # TODO: This would unreasonably show the 0th class label
            #  in visualization
            results.labels = results.scores.new_zeros(
                len(results), dtype=torch.long)
            del results.level_ids
        else:
            # To avoid some potential error
            results_ = InstanceData()
            if isinstance(results.bboxes, BaseBoxes):
                results_.bboxes = results.bboxes.empty_boxes()
            else:
                results_.bboxes = results.scores.new_zeros(0, 4)
            results_.scores = results.scores.new_zeros(0)
            results_.labels = results.scores.new_zeros(0)
            results = results_
        return results
