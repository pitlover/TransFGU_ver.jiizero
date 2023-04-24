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
                img: torch.Tensor,
                label: torch.Tensor = None,
                pseudo_things: torch.Tensor = None,
                pseudo_stuffs: torch.Tensor = None,
                is_augment: bool = False,
                epoch: int = 200,
                iter: int = 0
                ):
        feat, results = self.model(img=img, label=label, pseudo_things=pseudo_things, pseudo_stuffs=pseudo_stuffs,
                                   is_augment=is_augment, epoch=epoch, iter=iter)

        if self.training:
            model_loss = self.cat_weight * results["cat-loss"] + self.uncertainty_weight * results[
                "uncertainty-loss"] + self.cls_emb_weight * results["cls-emb-loss"]

            results["loss"] = model_loss

        return feat, results
