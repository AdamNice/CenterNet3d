import torch
from torch.nn import functional as F
from torch import nn
from mmdet.models import HEADS
from ..activations import _sigmoid,Mish
from collections import defaultdict

import torch
from torch import nn as nn

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from mmdet.models import HEADS
from ..builder import build_loss

@HEADS.register_module()
class Center3DHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 train_cfg,
                 test_cfg,
                 activation="lrelu",
                 bbox_coder=dict(type="XYZWLHRBoxCoder",voxel_size=[0.05,0.05,0.1],
                                pc_range=[0, -40, -3, 70.4, 40, 1],num_rad_bins=12,
                                downsample_ratio=4.0,min_overlap=0.01),
                 loss_cls=dict(
                     type='ModifiedFocalLoss',loss_weight=0.5),
                 loss_xy=dict(
                     type='GatherBalancedL1Loss',loss_weight=1.0),
                 loss_z=dict(
                     type='GatherBalancedL1Loss', loss_weight=1.0),
                 loss_dim=dict(
                     type='GatherBalancedL1Loss', loss_weight=1.0),
                 loss_dir=dict(
                     type='GatherBinResLoss', loss_weight=1.0),
                 loss_corner=None,
                 loss_decode=None,
                 ):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(Center3DHead, self).__init__()
        self.num_class = num_classes
        self.in_channels=in_channels
        self.tran_cfg=train_cfg
        self.test_cfg=test_cfg
        self.corner_attention=False
        if loss_corner is not None:
            self.corner_attention=True

        self.activaton_fun = nn.ReLU(inplace=True)
        if activation == "lrelu":
            self.activaton_fun = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "mish":
            self.activaton_fun = Mish()
        # box_coder["num_class"]=num_class
        #build box coder
        self.box_coder=build_bbox_coder(bbox_coder)
        self.loss_cls=build_loss(loss_cls)
        self.loss_xy = build_loss(loss_xy)
        self.loss_z = build_loss(loss_z)
        self.loss_dim = build_loss(loss_dim)
        self.loss_dir = build_loss(loss_dir)
        self.loss_decode=self.loss_corner=None
        if loss_corner is not None:
            self.loss_corner=build_loss(loss_corner)
            print("use corner attention module!")
        if loss_decode is not None:
            loss_decode["box_coder"]=bbox_coder
            self.loss_decode=build_loss(loss_decode)
            print("use decode loss!")

        self.heads = {"cls_preds": 1,"xy_preds": 2, "z_preds": 1,"dim_preds": 3,  "dir_preds": 2}
        if bbox_coder["num_rad_bins"]>0:
            assert loss_dir["type"]=="GatherBinResLoss","num_rad_bin greater than 0, GatherBinResLoss is required"
            self.heads["dir_preds"]=bbox_coder["num_rad_bins"]*2
        if self.corner_attention:
            self.heads['corner_preds'] = 1

        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(in_channels, feat_channels,
                          kernel_size=3, padding=1, bias=True),
                self.activaton_fun,
                nn.Conv2d(feat_channels, classes,
                          kernel_size=1, stride=1, ))
            if head in ["cls_preds","corner_preds"]:
                fc[-1].bias.data.fill_(-7.94)
            self.__setattr__(head, fc)

    def forward(self, x):
        z = {}

        if not self.training and self.corner_attention:
            self.heads.pop('corner_preds')

        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
            if head in ["cls_preds","corner_preds","xy_preds"]:
                z[head]=_sigmoid(z[head])

        return z

    def init_weights(self):
        """Initialize the weights of head."""
        # bias_cls = bias_init_with_prob(0.01)
        # normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        # normal_init(self.conv_reg, std=0.01)
        pass

    def get_bboxes(self,pred_dicts,input_metas):
        """
        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            input_metas (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): Whether th rescale bbox.

        Returns:
            list[tuple]: Prediction resultes of batches.
        """

        return self.box_coder.decode_center(pred_dicts,input_metas,score_threshold=self.test_cfg['score_thr'])

    def loss(self,pred_dict, gt_labels,gt_bboxes):
        gt_dict=self.box_coder.generate_target(gt_labels,gt_bboxes)
        mask=gt_dict["reg_mask"]
        index=gt_dict["gt_index"]
        cls_loss=self.loss_cls(pred_dict["cls_preds"],gt_dict["score_map"])
        xy_loss=self.loss_xy(pred_dict["xy_preds"],mask,index,gt_dict["gt_xyz"][...,:2])
        z_loss=self.loss_z(pred_dict["z_preds"],mask,index,gt_dict["gt_xyz"][...,2:])
        dim_loss=self.loss_dim(pred_dict["dim_preds"],mask,index,gt_dict["gt_dim"])
        dir_loss=self.loss_dir(pred_dict["dir_preds"],mask,index,gt_dict["gt_dir"])
        # total_loss = cls_loss + xy_loss + z_loss + dim_loss + dir_loss
        loss_dict = {
            "cls_loss": cls_loss,
            "xy_loss": xy_loss,
            "z_loss": z_loss,
            "dim_loss": dim_loss,
            "dir_loss": dir_loss,
            # "total_loss":total_loss
        }
        if self.loss_corner is not None:
            corner_loss=self.loss_corner(pred_dict["corner_preds"],gt_dict["corner_map"])
            loss_dict["corner_loss"]=corner_loss
            # loss_dict["total_loss"]=loss_dict["total_loss"]+corner_loss
        if self.loss_decode is not None:
            decode_loss=self.loss_decode(pred_dict,mask,index,gt_dict["gt_boxes3d"],gt_dict["gt_dir"])
            loss_dict["decode_loss"]=decode_loss
            # loss_dict["total_loss"]=loss_dict["total_loss"]+decode_loss

        return loss_dict

@HEADS.register_module()
class Center3DHeadDepthAware(Center3DHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 train_cfg,
                 test_cfg,
                 activation="lrelu",
                 bbox_coder=dict(type="XYZWLHRBoxCoder", voxel_size=[0.05, 0.05, 0.1],
                                pc_range=[0, -40, -3, 70.4, 40, 1], num_rad_bins=12,
                                downsample_ratio=4.0, min_overlap=0.01),
                 loss_cls=dict(
                     type='ModifiedFocalLoss', loss_weight=0.5),
                 loss_xy=dict(
                     type='GatherBalancedL1Loss', loss_weight=1.0),
                 loss_z=dict(
                     type='GatherBalancedL1Loss', loss_weight=1.0),
                 loss_dim=dict(
                     type='GatherBalancedL1Loss', loss_weight=1.0),
                 loss_dir=dict(
                     type='GatherBinResLoss', loss_weight=1.0),
                 loss_corner=None,
                 loss_decode=None):
        super(Center3DHeadDepthAware, self).__init__(
            num_classes,
            in_channels,
            feat_channels,
            train_cfg,
            test_cfg,
            activation=activation,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_xy=loss_xy,
            loss_z=loss_z,
            loss_dim=loss_dim,
            loss_dir=loss_dir,
            loss_corner=loss_corner,
            loss_decode=loss_decode
        )
        self.heads = {'xy_preds': 2, 'dim_preds': 3, 'z_preds': 1, 'dir_preds': 2, 'cls_preds': 1}
        self.depths = ["easy", "mod", "hard"]

        if bbox_coder['num_rad_bins']>0:
            assert loss_dir["type"] == "GatherBinResLoss", "num_rad_bin greater than 0, GatherBinResLoss is required"
            self.heads['dir_preds']=bbox_coder["num_rad_bins"]*2
        if self.corner_attention:
            self.heads['corner_preds']=1

        for difficulty in self.depths:
            for head,channels in self.heads.items():
                difficult_head=difficulty+"_"+head
                fc = nn.Sequential(
                    nn.Conv2d(in_channels, feat_channels,
                              kernel_size=3, padding=1, bias=True),
                    self.activaton_fun,
                    nn.Conv2d(feat_channels, channels,
                              kernel_size=1, stride=1,))
                if 'corner_preds' in head or 'cls_preds' in head:
                    fc[-1].bias.data.fill_(-7.94)
                self.__setattr__(difficult_head, fc)


        if not self.training and self.corner_attention:
            self.heads.pop('corner_preds')

    def forward(self, x):
        z = defaultdict(list)
        out={}
        easy=x[...,:118]
        mod=x[...,118:235]
        hard=x[...,235:]

        depth_fps={self.depths[0]:easy,
                   self.depths[1]:mod,
                   self.depths[2]:hard}

        for difficulty in self.depths:
            for head in self.heads:
                difficult_head=difficulty+"_"+head
                z[head].append(self.__getattr__(difficult_head)(depth_fps[difficulty]))

        for key,elems in z.items():
            out[key]=torch.cat(elems,dim=-1)
            if key in ["cls_preds","corner_preds","xy_preds"]:
               out[key]=_sigmoid(out[key])

        return out