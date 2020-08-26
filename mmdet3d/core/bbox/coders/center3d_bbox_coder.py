from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core import multi_apply
import torch.nn.functional as F

@BBOX_CODERS.register_module()
class Center3DBoxCoder(BaseBBoxCoder):
    def __init__(self,num_class=1,voxel_size=[0.05,0.05,0.1],
                                pc_range=[0, -40, -3, 70.4, 40, 1],num_rad_bins=12,
                                downsample_ratio=4.0,min_overlap=0.01):
        super(Center3DBoxCoder,self).__init__()
        self.num_class=num_class
        self.voxel_size=torch.tensor(voxel_size).cuda()
        self.pc_range=torch.tensor(pc_range).cuda()
        self.num_rad_bin=num_rad_bins
        self.downsample_ratio=downsample_ratio
        self.min_overlap=min_overlap
        self.max_gt_num=50
        self.grid_size=torch.round(((self.pc_range[3:]-self.pc_range[:3])/self.voxel_size)).to(torch.long)
        self.output_size=(self.grid_size[:2]//downsample_ratio).to(torch.int)[[1,0]]

    def generate_target_single(self,gt_labels,gt_bboxes):

        # valid_mask=gt_labels>=0
        # gt_labels=gt_labels[valid_mask]
        # print("gt labels is ",gt_labels)
        # gt_bboxes=gt_bboxes[valid_mask]
        boxes=gt_bboxes.tensor.cuda()
        boxes_center=boxes[:,[0,1]]
        # classes=gt_labels
        num_boxes=boxes.shape[0]
        # init gt tensors
        gt_scoremap = gt_labels.new_zeros((self.num_class,*self.output_size )).float()
        gt_corner_scoremap = gt_labels.new_zeros((self.num_class, *self.output_size)).float()
        gt_xyz = gt_labels.new_zeros((self.max_gt_num,3)).float()
        gt_dim = gt_labels.new_zeros((self.max_gt_num,3)).float()
        gt_dir = gt_labels.new_zeros((self.max_gt_num,2)).float()
        reg_mask = gt_labels.new_zeros((self.max_gt_num)).long()
        gt_index = gt_labels.new_zeros((self.max_gt_num)).long()
        gt_boxes3d=gt_labels.new_zeros((self.max_gt_num,7)).float()

        boxes_dim2d_scaled = boxes[:, 3:5] /self.voxel_size[:2]/self.downsample_ratio
        boxes_center=(boxes_center-self.pc_range[:2])/self.voxel_size[:2]/self.downsample_ratio
        boxes_center_int=boxes_center.to(torch.int)
        Center3DBoxCoder.generate_score_map(
            gt_scoremap, gt_labels, boxes_dim2d_scaled, boxes_center_int, self.min_overlap)

        # box_instances=LiDARInstance3DBoxes(gt_bboxes,origin=(0.5, 0.5, 0))
        gt_corners_3d = gt_bboxes.corners.cuda()
        gt_corners_2d = gt_corners_3d[:, ::2, :2]  # (num_boxes,4,2)
        gt_corners_2d = gt_corners_2d.reshape(-1, 2)
        gt_corners_2d = (gt_corners_2d - self.pc_range[:2]) / self.voxel_size[:2] / self.downsample_ratio
        gt_corners_int = gt_corners_2d.to(torch.int32)
        cls_corners = gt_labels.repeat_interleave(4,dim=0)
        Center3DBoxCoder.generate_score_map(
            gt_corner_scoremap, cls_corners, boxes_dim2d_scaled.repeat_interleave(4,dim=0), gt_corners_int, self.min_overlap)


        gt_index[:num_boxes] = boxes_center_int[:, 1] * self.output_size[1] + boxes_center_int[:, 0]
        gt_xyz[:num_boxes,:2] = boxes_center-boxes_center_int
        gt_xyz[:num_boxes,2]=boxes[:,2]
        reg_mask[:num_boxes] = 1
        gt_dim[:num_boxes]=boxes[:,3:6]
        gt_boxes3d[:num_boxes]=boxes

        if self.num_rad_bin<=0:
            gt_dir[:num_boxes,0]=torch.sin(boxes[:,6])
            gt_dir[:num_boxes,1]=torch.cos(boxes[:,6])
        else:
            angle_per_class = (2 * np.pi)/self.num_rad_bin
            heading_angle =boxes[:,6] % (2 * np.pi)  # 0 ~ 2pi
            shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
            ry_bin_label = torch.floor(shift_angle / angle_per_class).to(torch.long)
            gt_dir[:num_boxes,0]=ry_bin_label
            ry_res_label = shift_angle - (ry_bin_label.to(torch.float) * angle_per_class + angle_per_class / 2)
            ry_res_norm_label = ry_res_label / (angle_per_class / 2)
            gt_dir[:num_boxes,1]=ry_res_norm_label



        return (gt_scoremap,gt_xyz,gt_dim,gt_dir,gt_index,reg_mask,gt_corner_scoremap,gt_boxes3d)

    def generate_target(self, gt_labels_list, gt_bboxes_list):

        (all_gt_scoremap,all_gt_xyz,all_gt_dim,all_gt_dir,
         all_gt_index,all_reg_mask,all_gt_corner_scoremap,all_gt_boxes3d)= multi_apply(self.generate_target_single,gt_labels_list,gt_bboxes_list)

        gt_dict = {
            "score_map": torch.stack(all_gt_scoremap,dim=0),  # (num_cls,H,W)
            "gt_xyz": torch.stack(all_gt_xyz,dim=0),
            "gt_dim": torch.stack(all_gt_dim,dim=0),
            "gt_dir": torch.stack(all_gt_dir,dim=0),
            "gt_index":torch.stack(all_gt_index,dim=0),
            "reg_mask": torch.stack(all_reg_mask,dim=0),
            "corner_map":torch.stack(all_gt_corner_scoremap,dim=0),
            "gt_boxes3d":torch.stack(all_gt_boxes3d,dim=0)
        }

        return gt_dict

    def encode(self, bboxes, gt_bboxes):
        pass
    def decode(self, bboxes, bboxes_pred):
        pass

    def decode_center(self,
               pred_dict, img_metas,
               K=50, score_threshold=0.25):
        r"""
        decode output feature map to detection results
        """

        fmap = pred_dict['cls_preds']

        dim_pred = pred_dict['dim_preds']
        xy_pred = pred_dict['xy_preds']
        z_pred = pred_dict['z_preds']
        dir_pred = pred_dict['dir_preds']
        batch, channel, height, width = fmap.shape

        # assert batch == 1, "only batch size 1 is supported"

        # image_ids = example['image_idx']

        fmap = self.pseudo_nms(fmap)

        scores, index, clses, ys, xs = self.topk_score(fmap, K=K)

        xy_pred = gather_feature(xy_pred, index, use_transform=True)
        xy_pred = xy_pred.reshape(batch, K, 2)
        xs = xs.view(batch, K, 1) + xy_pred[:, :, 0:1]
        ys = ys.view(batch, K, 1) + xy_pred[:, :, 1:2]
        xs = xs * self.downsample_ratio * self.voxel_size[0] + self.pc_range[0]
        ys = ys * self.downsample_ratio* self.voxel_size[1] + self.pc_range[1]

        z_pred = gather_feature(z_pred, index, use_transform=True)
        zs = z_pred

        dim_pred = gather_feature(dim_pred, index, use_transform=True)
        dim_pred = dim_pred.reshape(batch, K, 3)

        clses_batch = clses.reshape(batch, K).float().detach()
        scores_batch = scores.reshape(batch, K).detach()

        dir_pred = gather_feature(dir_pred, index, use_transform=True)
        if self.num_rad_bin<=0:
            dir_pred = torch.atan2(dir_pred[:, :, 0], dir_pred[:, :, 1])
            dir_pred = dir_pred.reshape(batch, K, 1)

        else:
            dir_bin = torch.argmax(dir_pred[:, :, :self.num_rad_bin], dim=-1)
            dir_res_norm = torch.gather(dir_pred[:, :, self.num_rad_bin:], dim=-1,
                                        index=dir_bin.unsqueeze(dim=-1)).squeeze(dim=-1)
            angle_per_class = (2 * np.pi) / self.num_rad_bin
            dir_res = dir_res_norm * (angle_per_class / 2)
            dir_pred = (dir_bin.float() * angle_per_class + dir_res)% (2 * np.pi)
            dir_pred[dir_pred > np.pi] -= 2 * np.pi
            dir_pred+=np.pi
            dir_pred = dir_pred.reshape(batch, K, 1)

        bboxes_batch = torch.cat([xs, ys, zs, dim_pred, dir_pred], dim=-1).detach()

        detections=[]
        for clses,scores, bboxes,img_meta in zip(clses_batch,scores_batch, bboxes_batch,img_metas):
            keep = scores > score_threshold
            scores = scores[keep]
            clses = clses[keep]
            bboxes = bboxes[keep]
            bboxes = img_meta['box_type_3d'](bboxes, box_dim=7)
            detection = (bboxes, scores, clses, img_meta)
            detections.append(detection)

        return detections


    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap):
        radius = Center3DBoxCoder.get_gaussian_radius(gt_wh, min_overlap)
        # np.set_printoptions(suppress=True)
        # print("radius is ",radius)
        radius = torch.clamp(radius, min=0)
        radius = radius.to(torch.int)
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            Center3DBoxCoder.draw_gaussian(fmap[channel_index], centers_int[i], radius[i].item())


    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        box_tensor = box_size
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        # r1  = (b1 + sq1) / 2
        r1 = (b1 - sq1) / (2 * a1)

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 =torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        # r2  = (b2 + sq2) / 2
        r2 = (b2 - sq2) / (2 * a2)

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        # r3  = (b3 + sq3) / 2
        r3 = (b3 + sq3) / (2 * a3)

        return torch.min(r1, torch.min(r2, r3))
        # return np.min(box_size,axis=1)/3*2

    @staticmethod
    def gaussian2D(radius, sigma=1,rot=None):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = Center3DBoxCoder.gaussian2D((radius, radius), sigma=diameter / 6,rot=None)
        gaussian = fmap.new_tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap  = fmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top:y + bottom, x - left:x + right] = masked_fmap
        # return fmap
    @staticmethod
    def bbox_areas(bboxes):
        areas=bboxes[:,3]*bboxes[:,4]
        return areas


    def pseudo_nms(self,fmap, pool_size=3):
        r"""
        apply max pooling to get the same effect of nms

        Args:
            fmap(Tensor): output tensor of previous step
            pool_size(int): size of max-pooling
        """
        pad = (pool_size - 1) // 2
        fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=pad)
        keep = (fmap_max == fmap).float()
        return fmap * keep


    def topk_score(self,scores, K=40):
        """
        get top K point in score map
        """
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = (index / K).int()
        topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
        topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index  = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap