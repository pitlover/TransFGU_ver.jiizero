from typing import Dict, Tuple, List, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from copy import deepcopy
from utils.transfgu_utils import RandomApply, load_pretrained_weights

from model.transformers.VisionTransformer import VisionTransformer as vit
from model.transformers.MaskTransformer import MaskTransformer as Decoder


class TransFGU(nn.Module):
    def __init__(self, cfg: Dict, num_classes: int):
        super().__init__()
        # cfg = cfg["model"]

        self.n_things = cfg["decoder"]["n_things"]
        self.n_stuff = cfg["decoder"]["n_stuff"]
        self.n_cls = cfg["decoder"]["n_things"] + cfg["decoder"]["n_stuff"]
        self.n_cls_gt = cfg["decoder"]["n_things"] + cfg["decoder"]["n_stuff"]
        self.encoder = vit.__dict__[cfg["encoder"]["arch"]](patch_size=cfg["encoder"]["patch_size"], num_classes=0)
        self.encoder_teacher = vit.__dict__[cfg["encoder"]["arch"]](patch_size=cfg["encoder"]["patch_size"],
                                                                    num_classes=0)
        print(f"encoder {cfg['encoder']['arch']} {cfg['encoder']['patch_size']}x{cfg['encoder']['patch_size']} built.")
        load_pretrained_weights(self.encoder, cfg["pretrained_weight"], cfg["encoder"]["arch"],
                                cfg["encoder"]["patch_size"])
        self.decoder = Decoder(n_cls=self.n_cls, patch_size=cfg["encoder"]["patch_size"])
        self.decoder_teacher = Decoder(n_cls=self.n_cls, patch_size=cfg["encoder"]["patch_size"])

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

    def forward(self):
        return
