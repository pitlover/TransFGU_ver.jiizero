from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
from torchvision import transforms as T
import torch.nn.functional as F

from model.others.loss import cls_emb_loss, bootstrap_loss
from copy import deepcopy
from utils.transfgu_utils import RandomApply, load_pretrained_weights

from model.transformers.VisionTransformer import VisionTransformer as vit
from model.transformers.MaskTransformer import MaskTransformer as Decoder
from mmcv.ops.roi_align import roi_align


class TransFGU(nn.Module):
    def __init__(self, cfg: Dict, num_classes: int):
        super().__init__()
        # cfg = cfg["model"]

        # ----------- Option ---------- #
        self.bootstrapping_epoch = cfg["bootstrapping_epoch"]
        self.intervals = cfg["intervals"]
        self.is_bootstrapping = False  # Placeholder

        self.n_thing = cfg["decoder"]["n_thing"]
        self.n_stuff = cfg["decoder"]["n_stuff"]
        self.n_cls = cfg["decoder"]["n_thing"] + cfg["decoder"]["n_stuff"]
        self.n_cls_gt = cfg["decoder"]["n_thing"] + cfg["decoder"]["n_stuff"]
        self.patch_size = cfg["encoder"]["patch_size"]

        # ----------- Model ---------- #
        self.encoder = vit.__dict__[cfg["encoder"]["arch"]](patch_size=self.patch_size, num_classes=0)
        self.encoder_teacher = vit.__dict__[cfg["encoder"]["arch"]](patch_size=self.patch_size,
                                                                    num_classes=0)
        print(f"encoder {cfg['encoder']['arch']} {self.patch_size}x{self.patch_size} built.")
        load_pretrained_weights(self.encoder, cfg["pretrained_weight"], cfg["encoder"]["arch"],
                                self.patch_size)
        self.decoder = Decoder(n_cls=self.n_cls, patch_size=self.patch_size)
        self.decoder_teacher = Decoder(n_cls=self.n_cls, patch_size=self.patch_size)

        self.fix_encoder = cfg["encoder"]

        self.encoder_teacher.load_state_dict(deepcopy(self.encoder.state_dict()))
        self.decoder_teacher.load_state_dict(deepcopy(self.decoder.state_dict()))
        self.encoder_teacher.eval()
        self.decoder_teacher.eval()

        self.img_aug = torch.nn.Sequential(
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            T.RandomGrayscale(p=0.2),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
            T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.encoder_init_weights = deepcopy(self.encoder.state_dict())
        self.decoder_init_weights = deepcopy(self.decoder.state_dict())

        # ----------- Loss ---------- #
        self.cls_emb_loss = cls_emb_loss()
        self.bootstrap_loss = bootstrap_loss()

    def copy_and_reset(self):
        self.encoder_teacher.load_state_dict(self.encoder.state_dict())
        self.decoder_teacher.load_state_dict(self.decoder.state_dict())
        self.encoder.load_state_dict(self.encoder_init_weights)
        self.decoder.load_state_dict(self.decoder_init_weights)

    def data_augment(self, img, label, pseudo_label):
        device = img.device
        N, _, H, W = img.shape
        h, w = pseudo_label.shape[-2:]
        H_target, W_target = H // 2, W // 2
        h_pseudo_label, w_pseudo_label = H_target // self.patch_size, W_target // self.patch_size
        img_aug = self.img_aug(img / 255.)

        # ====== random resized crop ======
        scale_min, scale_max = 0.4, 1
        scales = torch.randint(int(scale_min * 10), int(scale_max * 10), (N, 1))[:, 0] / 10.
        crop_window_size_h, crop_window_size_w = (torch.tensor(H) * scales).int(), (torch.tensor(W) * scales).int()
        available_y, available_x = H - crop_window_size_h, W - crop_window_size_w

        y1, x1 = (available_y * torch.rand(N, 1)[:, 0]).int(), (available_x * torch.rand(N, 1)[:, 0]).int()

        img_crops = []
        label_crops = []
        for img_aug_, label_, y_, x_, crop_window_size_h_, crop_window_size_w_ in zip(img_aug, label, y1, x1,
                                                                                      crop_window_size_h,
                                                                                      crop_window_size_w):
            img_crops.append(F.interpolate(img_aug_[:, y_:y_ + crop_window_size_h_, x_:x_ + crop_window_size_w_][None],
                                           (H_target, W_target), mode='bilinear')[0])
            label_crops.append(
                F.interpolate(label_[y_:y_ + crop_window_size_h_, x_:x_ + crop_window_size_w_][None, None],
                              (H_target, W_target))[0, 0])
        img_crops = torch.stack(img_crops)
        label_crops = torch.stack(label_crops)

        rois = torch.stack([x1, y1, x1 + crop_window_size_w, y1 + crop_window_size_h], 1) * (h / H)
        rois = torch.cat([torch.range(0, N - 1)[:, None], rois], dim=1).to(device)
        pseudo_label_crop = roi_align(pseudo_label, rois, (h_pseudo_label, w_pseudo_label),
                                      1.0, 0, 'avg', True).squeeze(1)

        # ====== random flip ======
        flag = torch.rand(N)
        for n in range(N):
            if flag[n] < 0.5:
                img_crops[n] = img_crops[n][:, :, range(img_crops.shape[-1] - 1, -1, -1)]
                label_crops[n] = label_crops[n][:, range(label_crops.shape[-1] - 1, -1, -1)]
                pseudo_label_crop[n] = pseudo_label_crop[n][:, :, range(pseudo_label_crop.shape[-1] - 1, -1, -1)]

        return img_crops, label_crops, pseudo_label_crop

    def forward(self, img: torch.Tensor,
                label: torch.Tensor,
                pseudo_things: torch.Tensor,
                pseudo_stuffs: torch.Tensor,
                is_augment: bool = False,
                epoch: int = -1,
                iter: int = -1):
        '''

        :param img: (b, 3, h, w)
        :param label: (b, h, w)
        :param pseudo_things: (b, 12, h//8, w//8)
        :param pseudo_stuffs: (b, 15, h//8, w//8)
        '''
        if not epoch < self.bootstrapping_epoch and epoch % self.intervals == 0:
            self.copy_and_reset()

        pseudo_labels = torch.cat([pseudo_things, pseudo_stuffs], dim=1)

        is_bootstrapping = True if not epoch < self.bootstrapping_epoch else False

        if is_augment:
            img, label, pseudo_labels = self.data_augment(img, label, pseudo_labels)

        N, _, H, W = img.shape
        h, w = H // self.patch_size, W // self.patch_size
        N_cls_fgbg = self.n_cls
        th = 0.1

        with torch.no_grad():
            y, attn = self.encoder.forward_feat_attn(img)

        masks_cls, cls_embs = self.decoder(y[:, 1:], ret_cls_emb=True)
        masks_cls = masks_cls.transpose(1, 2).view(N, N_cls_fgbg, h, w).contiguous()

        output = dict()

        if self.trainig:
            loss_cls_emb = self.cls_emb_loss(cls_embs)

            if is_bootstrapping:
                with torch.no_grad():
                    self.decoder_teacher.eval()
                    self.encoder_teacher.eval()
                    y_teacher, _ = self.encoder_teacher.forward_feat_attn(img)
                    pseudo_labels_teacher = self.decoder_teacher(y_teacher[:, 1:])
                    pseudo_labels_teacher = pseudo_labels_teacher.transpose(1, 2).view(N, N_cls_fgbg, h, w).contiguous()
            else:
                pseudo_labels_teacher = None

            if not pseudo_labels.shape[-2:] == (h, w):
                pseudo_labels = F.interpolate(pseudo_labels, size=(h, w), mode='bilinear', align_corners=False)
            loss_cat, loss_uncertainty, bootstrapped_pseudo_labels = \
                self.bootstrap_loss(masks_cls, pseudo_label=pseudo_labels,
                                    pseudo_label_teacher=pseudo_labels_teacher, epoch=epoch)
            output["cat-loss"] = loss_cat.mean()
            output["uncertainty-loss"] = loss_uncertainty.mean()
            output["cls-emb-loss"] = loss_cls_emb.mean()

        return masks_cls, output
