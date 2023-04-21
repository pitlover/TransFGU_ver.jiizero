from typing import Dict, Tuple, List
import torch
import torch.nn as nn

from model.transfgu import TransFGU

__all__ = [
    "TransFGUWrapper"
]


class TransFGUWrapper(nn.Module):

    def __init__(self,
                 cfg: Dict,
                 loss_cfg: Dict,
                 model: TransFGU
                 ) -> None:
        super().__init__()
        # cfg = cfg
        # loss_cfg = cfg["loss"]

        self.cat_weight = loss_cfg["cat_weight"]
        self.uncertainty_weight = loss_cfg["uncertainty_weight"]
        self.cls_emb_weight = loss_cfg["cls_emb_weight"]

        self.model = model

    def forward(self,
                img1: torch.Tensor,
                label: torch.Tensor,
                img2: torch.Tensor = None,
                aug_weak: torch.Tensor = None,
                aug_strong: torch.Tensor = None,
                iter: int = 0,
                max_iter: int = 0):
        '''

        :param img1: (b, 3, h, w)
        :param label: (b * label_ratio, 3, h, w)
        :param img2:
        :param aug_weak:
        :param aug_strong:
        :param iter:
        :return:
        '''

        feat, results = self.model(img=img1, label=label, aug_weak=img2, aug_strong=aug_strong,
                                   iter=iter)
        if not self.training:
            return feat

        model_loss = self.cat_weight * results["cat-loss"] + self.uncertainty_weight * results[
            "uncertainty-loss"] + self.cls_emb_weight * results["cls-emb-loss"]

        results["loss"] = model_loss

        return feat, results
