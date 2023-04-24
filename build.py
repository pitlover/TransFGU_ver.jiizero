from typing import Dict, List
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.cuda.amp import GradScaler

from utils.dist_utils import is_distributed_set
from data.dataset import UnsegDataset
from model.transfgu import TransFGU
from wrapper.TransFGUWrapper import TransFGUWrapper


def build_model(cfg: Dict) -> nn.Module:
    # cfg
    model_name = cfg["name"].lower()

    if "transfgu" in model_name:
        model = TransFGUWrapper(cfg, cfg["loss"],
                                TransFGU(cfg["model"], num_classes=cfg["dataset"]["num_class"]))
    else:
        raise ValueError(f"Unsupported model type {model_name}.")

    return model


def split_paramaters(model: nn.Module, cfg: Dict):
    decoder_regularized = []
    decoder_not_regularized = []
    encoder_regularized = []
    encoder_not_regularized = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            if 'encoder' in name and not cfg["model"]["encoder"]["fix"]:
                encoder_not_regularized.append(param)
            elif 'decoder' in name:
                decoder_not_regularized.append(param)
            else:
                pass
        else:
            if 'encoder' in name and not cfg["model"]["encoder"]["fix"]:
                encoder_regularized.append(param)
            elif 'decoder' in name:
                decoder_regularized.append(param)
            else:
                pass
    return [
        {'params': encoder_regularized, 'lr': cfg["lr"] * cfg["lr_scale"]},
        {'params': encoder_not_regularized, 'weight_decay': 0., 'lr': cfg["lr"] * cfg["lr_scale"]},
        {'params': decoder_regularized},
        {'params': decoder_not_regularized, 'weight_decay': 0.},
    ]


def build_dataset(data_dir: str, is_train: bool, seed: int, cfg: Dict) -> UnsegDataset:
    # cfg = cfg["dataset"]
    dataset = UnsegDataset(
        dataset_name=cfg["name"].lower(),
        data_dir=data_dir,
        is_train=is_train,
        seed=seed,
        cfg=cfg
    )

    return dataset


def build_dataloader(dataset: UnsegDataset, batch_size: int, is_train: bool, cfg: Dict) -> DataLoader:
    if is_train:
        if is_distributed_set():
            sampler = DistributedSampler(dataset, shuffle=True, seed=0, drop_last=True)
            shuffle, drop_last = False, False
        else:
            sampler = None
            shuffle, drop_last = True, True
    else:
        if is_distributed_set():
            sampler = DistributedSampler(dataset, shuffle=False, seed=0, drop_last=False)
            shuffle, drop_last = False, False
        else:
            sampler = None
            shuffle, drop_last = False, False

    # When using DistributedSampler, don't forget to call dataloader.sampler.set_epoch(epoch)

    kwargs = dict(
        batch_size=batch_size,  # per-process (=per-GPU)
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.get("num_workers", 1),  # per-process
        # collate_fn=ImageNet.fast_collate_imagenet,
        pin_memory=True,
        drop_last=drop_last,
        prefetch_factor=4,
        persistent_workers=True
    )
    dataloader = DataLoader(dataset, **kwargs)
    return dataloader


def build_optimizer(model: nn.Module, cfg: Dict) -> SGD:
    params_opt = split_paramaters(model, cfg)
    optimizer_type = cfg["name"].lower()

    if optimizer_type == "sgd":
        optimizer = SGD(
            params_opt,
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg.get("weight_decay", 0.0),
            nesterov=False
        )
    elif optimizer_type == "adam":
        optimizer = Adam(params_opt,
                         lr=cfg["lr"],
                         weight_decay=cfg["weight_decay"])
    elif optimizer_type == "adamw":
        optimizer = AdamW(params_opt,
                          lr=cfg["lr"],
                          betas=cfg.get("betas", (0.9, 0.999)),
                          weight_decay=cfg.get("weight_decay", 0.0))
    else:
        raise ValueError(f"Unsupported optimizer type {optimizer_type}.")

    return optimizer


def build_scheduler(optimizer: SGD, cfg: Dict, iter_per_epoch: int, num_epoch: int = 1, num_accum: int = 1):
    scheduler_type = cfg["name"].lower()
    iter_per_epoch = iter_per_epoch // num_accum  # actual update count
    if scheduler_type == "constant":
        scheduler = ConstantLR(optimizer,
                               factor=cfg.get("factor", 1.0),
                               total_iters=0)
    elif scheduler_type == "cos":
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=num_epoch,
                                      eta_min=cfg.get("min_lr", 0.0),
                                      last_epoch=-1)
    elif scheduler_type == "custom":
        warmup_cfg = cfg["warmup"]
        warmup = LinearLR(
            optimizer,
            start_factor=warmup_cfg["start_factor"],
            end_factor=1.0,
            total_iters=warmup_cfg["epochs"] * iter_per_epoch,
        )
        decay_cfg = cfg["decay"]
        decay = CosineAnnealingLR(
            optimizer,
            T_max=decay_cfg["epochs"] * iter_per_epoch,
            eta_min=decay_cfg["min_lr"],
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_cfg["epochs"] * iter_per_epoch]
        )
    else:
        raise ValueError(f"Unsupported optimizer type {scheduler_type}.")

    return scheduler


def build_scaler(is_fp16: bool = False) -> GradScaler:
    scaler = GradScaler(init_scale=2048, growth_interval=1000, enabled=is_fp16)
    return scaler
