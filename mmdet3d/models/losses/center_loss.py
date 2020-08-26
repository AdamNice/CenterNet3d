#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.core.bbox.coders import gather_feature
from mmdet.models.builder import LOSSES
from mmdet.core import build_bbox_coder

@LOSSES.register_module()
class ModifiedFocalLoss(nn.Module):
    def __init__(self,loss_weight,reduction="mean"):
        super(ModifiedFocalLoss,self).__init__()
        self.weight=loss_weight
        self.reduction=reduction

    def forward(self,pred,target):
        loss=modified_focal_loss(pred,target,reduction=self.reduction)
        loss=loss*self.weight
        return loss

@LOSSES.register_module()
class GatherBalancedL1Loss(nn.Module):
    def __init__(self,loss_weight,beta=1.0, alpha=0.5, gamma=1.5,reduction="none"):
        super(GatherBalancedL1Loss,self).__init__()
        self.beta=beta
        self.alpha=alpha
        self.gamma=gamma
        self.weight=loss_weight
        self.reduction=reduction
        assert reduction=="none","only none reduction is support!"

    def forward(self,output,mask,index,target):
        pred = gather_feature(output, index, use_transform=True)  # (-1,C)
        mask = mask.unsqueeze(dim=2).expand_as(pred).float()
        pred=pred*mask
        target=target*mask

        assert pred.size() == target.size() and target.numel() > 0
        loss=balanced_l1_loss(pred,target,beta=self.beta,alpha=self.alpha,gamma=self.gamma,reduction=self.reduction)
        loss = loss.sum() / (mask.sum() + 1e-4)*self.weight
        return loss

@LOSSES.register_module()
class GatherL1Loss(nn.Module):
    def __init__(self,loss_weight,reduction="none"):
        super(GatherL1Loss,self).__init__()
        self.weight=loss_weight
        self.reduction=reduction
        assert reduction=="none","only none reduction is support!"
    def forward(self,output,mask,index,target):
        pred = gather_feature(output, index, use_transform=True)  # (-1,C)
        mask = mask.unsqueeze(dim=2).expand_as(pred).float()
        pred=pred*mask
        target=target*mask
        assert pred.size() == target.size() and target.numel() > 0
        loss = F.l1_loss(pred * mask, target * mask, reduction=self.reduction)
        loss = loss / (mask.sum() + 1e-4)*self.weight
        return loss

@LOSSES.register_module()
class GatherBinResLoss(nn.Module):
    def __init__(self,loss_weight,num_rad_bin=12,reduction="none"):
        super(GatherBinResLoss,self).__init__()
        self.weight=loss_weight
        self.reduction=reduction
        self.num_rad_bin=num_rad_bin

    def dir_bin_res_loss(self,dir_preds,mask,index,gt_dir):
        preds = gather_feature(dir_preds, index, use_transform=True)  # (B,-1,C)

        pred_bin=preds[...,:self.num_rad_bin]
        pred_reg=preds[...,self.num_rad_bin:]

        gt_bin=gt_dir[...,0]
        gt_bin=gt_bin.long()
        gt_reg=gt_dir[...,1]
        mask = mask.unsqueeze(dim=2).expand_as(pred_bin).float()
        ry_bin_onehot = torch.cuda.FloatTensor(gt_bin.size(0),gt_bin.size(1),self.num_rad_bin).zero_()
        ry_bin_onehot.scatter_(2, gt_bin.unsqueeze(-1), 1)
        loss_ry_bin = F.cross_entropy((pred_bin*mask).view(-1,pred_bin.size(-1)),
                                      (gt_bin*(mask[...,0].long())).view(-1),reduction='sum')
        loss_ry_res = F.smooth_l1_loss((pred_reg * mask*ry_bin_onehot).sum(dim=-1),
                                       gt_reg*mask[...,0],reduction='sum')
        loss_ry_res = loss_ry_res / (mask[...,0].sum() + 1e-4)
        loss_ry_bin = loss_ry_bin / (mask.sum() + 1e-4)
        return loss_ry_bin+loss_ry_res


    def forward(self,dir_preds,mask,index,gt_dir):
        loss=self.dir_bin_res_loss(dir_preds,mask,index,gt_dir)
        return loss*self.weight

@LOSSES.register_module()
class Boxes3dDecodeLoss(nn.Module):
    def __init__(self,loss_weight,box_coder=None):
        super(Boxes3dDecodeLoss,self).__init__()
        self.weight=loss_weight
        # assert loss_type in ["smooth_l1","l1","balanced_l1"],"loss type {} is not support".format(loss_type)
        # if loss_type=="smooth_l1":
        #     self.loss_fun=smooth_l1_loss
        # elif loss_type=="l1":
        #     self.loss_fun=F.l1_loss
        # elif loss_type=="balanced_l1":
        #     self.loss_fun=balanced_l1_loss
        # self.box_coder=box_coder
        self.box_coder = build_bbox_coder(box_coder)

    def forward(self,pred_dict, mask, index, target,gt_dir=None):
        #

        # fmap=example['score_map']
        # print("fmap shape is ",fmap.shape, fmap.dtype,fmap.device)
        voxel_size = self.box_coder.voxel_size
        pc_range = self.box_coder.pc_range
        fmap = pred_dict['cls_preds']
        batch,channels,height,width=fmap.shape
        dim_pred = pred_dict['dim_preds']
        dim_pred = gather_feature(dim_pred, index, use_transform=True)
        xy_pred = pred_dict['xy_preds']
        xy_pred = gather_feature(xy_pred, index, use_transform=True)
        z_pred = pred_dict['z_preds']
        z_pred = gather_feature(z_pred, index, use_transform=True)
        dir_pred = pred_dict['dir_preds']
        dir_pred = gather_feature(dir_pred, index, use_transform=True)

        if self.box_coder.num_rad_bin<=0:
            dir_pred=torch.atan2(dir_pred[:, :, 0:1], dir_pred[:, :, 1:])
        else:
            # assert gt_dir is not None,"dir_bin loss require gt_dir in decode loss"
            # angle_per_class = (2 * np.pi) / self.box_coder.num_rad_bin
            # gt_bin=gt_dir[:,:,0:1]
            # gt_res = gt_dir[:,:,1:2] * (angle_per_class / 2)
            # # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
            # dir_pred = (gt_bin.float() * angle_per_class + gt_res) % (2 * np.pi)
            # dir_pred[dir_pred > np.pi] -= 2 * np.pi
            angle_per_class = (2 * np.pi) / self.box_coder.num_rad_bin
            dir_bin = torch.argmax(dir_pred[:, :, :self.box_coder.num_rad_bin], dim=-1)
            dir_res_norm = torch.gather(dir_pred[:, :, self.box_coder.num_rad_bin:], dim=-1,
                                        index=dir_bin.unsqueeze(dim=-1)).squeeze(dim=-1)
            dir_res = dir_res_norm * (angle_per_class / 2)
            # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
            dir_pred = (dir_bin.float() * angle_per_class + dir_res) % (2 * np.pi)
            dir_pred[dir_pred > np.pi] -= 2 * np.pi
            dir_pred = dir_pred.unsqueeze(dim=-1)

        ys = (index / width).int().float().unsqueeze(-1)
        xs = (index % width).int().float().unsqueeze(-1)
        # xsys=torch.stack([xs,ys],dim=-1)

        xs = xs + xy_pred[:, :, 0:1]
        ys = ys + xy_pred[:, :, 1:2]
        xs = xs * self.box_coder.downsample_ratio * voxel_size[0] + pc_range[0]
        ys = ys * self.box_coder.downsample_ratio * voxel_size[1] + pc_range[1]


        boxes_pred=torch.cat([xs,ys,z_pred,dim_pred,dir_pred],dim=-1).reshape(-1,7)
        boxes_pred_instances=LiDARInstance3DBoxes(boxes_pred,origin=(0.5,0.5,0))
        corners_pred=boxes_pred_instances.corners.reshape(batch,-1,8,3)
        boxes_gt=target.reshape(-1,7)
        boxes_gt_instances=LiDARInstance3DBoxes(boxes_gt,origin=(0.5,0.5,0))
        corners_gt = boxes_gt_instances.corners.reshape(batch,-1,8,3)
        boxes_gt_flip=boxes_gt.clone()
        boxes_gt_flip[:,6]+=np.pi
        boxes_gt_flip_instances=LiDARInstance3DBoxes(boxes_gt_flip,origin=(0.5,0.5,0))
        corners_gt_flip=boxes_gt_flip_instances.corners.reshape(batch,-1,8,3)
        corners_dist = torch.min(torch.norm(corners_pred - corners_gt, dim=3),
                                torch.norm(corners_pred - corners_gt_flip, dim=3))
        mask = mask.unsqueeze(dim=-1).expand_as(corners_dist).float()
        corners_dist=corners_dist*mask

        def smooth_l1_loss(diff, beta):
            if beta < 1e-5:
                loss = torch.abs(diff)
            else:
                n = torch.abs(diff)
                loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)
            return loss

        # (N, 8)
        loss = smooth_l1_loss(corners_dist, beta=1.0)
        loss = loss.sum() / (mask.sum() + 1e-4)*self.weight
        return loss


def modified_focal_loss(pred, gt,reduction="sum"):
    """
    focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if reduction=="none":
        loss=pos_loss+neg_loss
    elif reduction=="sum":
        loss=pos_loss.sum()+neg_loss.sum()

    elif reduction=="mean":
        num_pos  = pos_inds.float().sum()
        if num_pos == 0:
            loss = neg_loss.sum()
        else:
            loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
    else:
        raise NotImplementedError
    return loss

def balanced_l1_loss(pred,target, beta=1.0, alpha=0.5, gamma=1.5,reduction="none"):

    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(
        diff < beta,
        alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta,)
    if reduction=="none":
        loss=loss
    elif reduction=="sum":
        loss=loss.sum()
    elif reduction=="mean":
        loss=loss.mean()
    else:
        raise NotImplementedError
    return loss

def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


def sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        reduction='sum')[None] / avg_factor


