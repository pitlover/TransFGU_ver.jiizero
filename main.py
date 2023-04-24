from typing import Dict
from collections import OrderedDict
import os
import time
import pprint

import wandb
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils.clip_grad import clip_grad_norm_

from utils.config_utils import prepare_config
from utils.wandb_utils import set_wandb
from utils.random_utils import set_seed
from utils.dist_utils import set_dist, is_distributed_set, is_master, barrier, get_world_size
from utils.dist_utils import all_reduce_dict
from utils.print_utils import time_log
from utils.param_utils import count_params, compute_param_norm

from model.others.metric import scores, UnsegMetrics
from build import (build_dataset, build_dataloader, build_model,
                   build_optimizer, build_scheduler, build_scaler)
from wrapper.TransFGUWrapper import TransFGUWrapper


def train_epoch(
        model: TransFGUWrapper,
        optimizer,
        scheduler,
        scaler,
        dataloader,
        cfg: Dict,
        device: torch.device,
        current_iter: int,
        epoch: int,
) -> int:
    print_interval = cfg["print_interval_iters"]
    fp16 = cfg.get("fp16", False)
    num_accum = cfg.get("num_accum", 1)
    clip_grad = cfg.get("clip_grad", 10.0)

    model.train()
    torch.set_grad_enabled(True)  # same as 'with torch.enable_grad():'
    grad_norm = torch.tensor(0.0, dtype=torch.float32, device=device)  # placeholder

    forward_time = 0.0
    backward_time = 0.0
    step_time = 0.0

    data_start_time = time.time()

    for it, data in enumerate(dataloader):
        s = time_log()
        s += f"Current iter: {current_iter} (epoch done: {it / len(dataloader) * 100:.2f} %)\n"

        # -------------------------------- data -------------------------------- #
        img, label, pseudo_things, pseudo_stuffs = data["img"], data["label"], data["pseudo_label_things"], \
                                                   data["pseudo_label_stuff"]
        img, label, pseudo_things, pseudo_stuffs = img.to(device, non_blocking=True), \
                                                   label.to(device, non_blocking=True), \
                                                   pseudo_things.to(device, non_blocking=True), \
                                                   pseudo_stuffs.to(device, non_blocking=True)

        data_time = time.time() - data_start_time

        # -------------------------------- loss -------------------------------- #
        if it % num_accum == 0:
            optimizer.zero_grad(set_to_none=True)

        if it % num_accum == (num_accum - 1):  # update step
            forward_start_time = time.time()
            with amp.autocast(enabled=fp16):
                _, output = model(img=img, label=label, pseudo_things=pseudo_things, pseudo_stuffs=pseudo_stuffs,
                                  is_augment=True,
                                  epoch=epoch, iter=it)  # {"loss", "acc1"}
            forward_time = time.time() - forward_start_time

            backward_start_time = time.time()
            loss = output["loss"]
            loss = loss / num_accum
            scaler.scale(loss).backward()
            backward_time = time.time() - backward_start_time

            step_start_time = time.time()
            # scaler.unscale_(optimizer)
            # grad_norm = clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()
            step_time = time.time() - step_start_time

        elif isinstance(model, DistributedDataParallel):  # non-update step and DDP
            with model.no_sync():
                with amp.autocast(enabled=fp16):
                    _, output = model(img=img, label=label, pseudo_things=pseudo_things, pseudo_stuffs=pseudo_stuffs,
                                      iter=it, max_iter=len(dataloader))  # {"loss", "acc1"}

                loss = output["loss"]
                loss = loss / num_accum
                scaler.scale(loss).backward()

        else:  # non-update step and not DDP
            with amp.autocast(enabled=fp16):
                _, output = model(img=img, label=label, pseudo_things=pseudo_things, pseudo_stuffs=pseudo_stuffs,
                                  iter=it, max_iter=len(dataloader))  # {"loss", "acc1"}

            loss = output["loss"]
            loss = loss / num_accum
            scaler.scale(loss).backward()

        # -------------------------------- print -------------------------------- #

        if (it > 0) and (it % print_interval == 0):
            output = all_reduce_dict(output, op="mean")
            param_norm = compute_param_norm(model.parameters())
            lr = scheduler.get_last_lr()[0]

            for k, v in output.items():
                s += f"... {k}: {v.item() if isinstance(v, torch.Tensor) else v:.6f}\n"
            s += f"... LR: {lr:.6f}\n"
            s += f"... grad/param norm: {grad_norm.item():.3f} / {param_norm.item():.3f}\n"
            s += f"... batch_size x num_accum x gpus = " \
                 f"{int(label.shape[0])} x {num_accum} x {get_world_size()}\n"
            s += f"... data/fwd/bwd/step time: " \
                 f"{data_time:.3f} / {forward_time:.3f} / {backward_time:.3f} / {step_time:.3f}"

            if is_master():
                print(s)
                log_dict = {
                    "grad_norm": grad_norm.item(),
                    "param_norm": param_norm.item(),
                    "lr": lr,
                    "iterations": current_iter,
                }
                for k, v in output.items():
                    log_dict[k] = v.item() if isinstance(v, torch.Tensor) else v
                wandb.log(log_dict)

        current_iter += 1
        data_start_time = time.time()

    return current_iter


def valid_epoch(
        model: TransFGUWrapper,
        dataloader,
        cfg: Dict,
        device: torch.device,
        current_iter: int,
) -> Dict:
    fp16 = cfg.get("fp16", False)

    model.eval()
    torch.set_grad_enabled(False)  # same as 'with torch.no_grad():'

    targets, preds = np.array([]), np.array([])
    output = {}
    n_cls = cfg["n_cls"]
    assert n_cls == 27
    histogram = np.zeros((n_cls, n_cls))

    for it, data in enumerate(dataloader):
        # -------------------------------- data -------------------------------- #
        img, label = data["img"], data["label"]
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        B, C, H, W = img.shape

        # -------------------------------- loss -------------------------------- #
        with amp.autocast(enabled=fp16):
            prob = model(img, iter=it)
            prob = F.interpolate(prob, size=(H, W), mode='bilinear', align_corners=False)
            preds = prob.topk(1, dim=1)[1].view(B, -1).cpu().numpy()
            label_cat_ = label.view(B, -1).cpu().numpy()
            histogram += scores(label_cat_, preds, n_cls)

    output = UnsegMetrics(histogram, n_cls)

    if is_master():
        wandb.log({
            "Cluster_mIoU": output["mean_iou"],
            "Cluster_Accuracy": output["overall_precision (pixel accuracy)"],
            "iterations": current_iter,
        })

    return output


def run(cfg: Dict, debug: bool = False, eval: bool = False) -> None:
    # ======================================================================================== #
    # Initialize
    # ======================================================================================== #
    device, local_rank = set_dist(device_type="cuda")

    if is_master():
        pprint.pprint(cfg)  # print config to check if all arguments are correctly given.

    save_dir = set_wandb(cfg, force_mode="disabled" if debug else None)
    set_seed(seed=cfg["seed"] + local_rank)

    # ======================================================================================== #
    # Data
    # ======================================================================================== #
    data_dir = cfg["data_dir"]

    train_dataset = build_dataset(data_dir, is_train=True, seed=cfg["seed"], cfg=cfg["dataset"])
    train_dataloader = build_dataloader(train_dataset, batch_size=cfg["dataloader"]["train"]["batch_size"],
                                        is_train=True, cfg=cfg["dataloader"]["train"])

    # Following previous works, we test unlabeled dataset in 'trainset'
    valid_dataset = build_dataset(data_dir, is_train=False, is_label=False, seed=cfg)
    valid_dataloader = build_dataloader(valid_dataset, batch_size=cfg["dataloader"]["valid"]["batch_size"],
                                        is_train=False, cfg=cfg["dataloader"]["valid"])

    # ======================================================================================== #
    # Model
    # ======================================================================================== #
    model = build_model(cfg)
    model = model.to(device)

    if is_distributed_set():
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=device,
                                        find_unused_parameters=True)  # TODO fix to freeze
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # optional for ImageNet
        model_m = model.module  # actual model without wrapping
    else:
        model_m = model

    if is_master():
        print(model)
        p1, p2 = count_params(model_m.parameters())
        print(f"Model parameters: {p1} tensors, {p2} elements.")

        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)

    if eval:
        if cfg["resume"]["checkpoint"] != None:
            ckpt = torch.load(cfg["resume"]["checkpoint"], map_location=device)
        else:
            raise ValueError(f"Eval mode requires checkpoint.")
    elif cfg["resume"]["checkpoint"] is not None:
        # checkpoint: {"model", "optimizer", "scheduler", "stats"}
        ckpt = torch.load(cfg["resume"]["checkpoint"], map_location=device)
    else:
        ckpt = None

    if ckpt is not None:
        model_m.load_state_dict(ckpt["model"], strict=cfg["resume"].get("strict", True))

    # ======================================================================================== #
    # Optimizer & Scheduler
    # ======================================================================================== #
    optimizer = build_optimizer(model_m, cfg=cfg["optimizer"])
    scheduler = build_scheduler(optimizer, cfg=cfg["scheduler"],
                                iter_per_epoch=len(train_dataset),
                                num_epoch=cfg["trainer"]["max_epochs"],
                                num_accum=cfg["trainer"].get("num_accum", 1))
    scaler = build_scaler(is_fp16=cfg["trainer"].get("fp16", False))

    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])

    # ======================================================================================== #
    # Trainer
    # ======================================================================================== #

    # -------- config -------- #
    train_cfg = cfg["trainer"]
    max_epochs = train_cfg["max_epochs"]
    valid_interval = train_cfg["valid_interval_epochs"]

    # -------- status -------- #
    current_epoch = 0
    current_iter = 0
    current_best_miou = 0.0
    best_epoch, best_iter = 0, 0

    if ckpt is not None:  # Eval
        current_epoch = ckpt["stats"]["epoch"]
        current_iter = ckpt["stats"]["iter"]
        current_best_miou = ckpt["stats"]["best_miou"]
        max_epochs = ckpt["stats"]["max_epochs"]

        # -------- check -------- #
        valid_result = valid_epoch(model, valid_dataloader, train_cfg, device, current_iter)
        s = time_log()
        s += f"Resume valid epoch {current_epoch} / {max_epochs}\n"
        s += f"... acc: {valid_result['acc']:.4f}\n"
        s += f"... miou: {valid_result['miou']:.4f}\n"
        if is_master():
            print(s)
        if eval:
            print("Best Eval Finish...")
            return
            # -------- main loop -------- #
    while current_epoch < max_epochs:
        if is_master():
            s = time_log()
            s += f"Start train epoch {current_epoch} / {max_epochs} (iter: {current_iter})"
            print(s)

        if is_distributed_set():
            # reset random seed of sampler, sampler should be DistributedSampler.
            train_label_dataloader.sampler.set_epoch(current_epoch)  # noqa
            train_unlabel_dataloader.sampler.set_epoch(current_epoch)  # noqa

        # -------- train body -------- #
        epoch_start_time = time.time()  # second
        current_iter = train_epoch(model, optimizer, scheduler, scaler, train_dataloader,
                                   train_cfg, device, current_iter, epoch=current_epoch)
        epoch_time = time.time() - epoch_start_time
        if is_master():
            s = time_log()
            s += f"End train epoch {current_epoch} / {max_epochs}, time: {epoch_time:.3f} s\n"
            # save checkpoint
            ckpt = OrderedDict()
            ckpt["model"] = model_m.state_dict()
            ckpt["optimizer"] = optimizer.state_dict()
            ckpt["scheduler"] = scheduler.state_dict()
            ckpt["scaler"] = scaler.state_dict()
            ckpt["stats"] = OrderedDict(epoch=current_epoch, iter=current_iter,
                                        best_miou=current_best_miou)
            torch.save(ckpt, os.path.join(save_dir, "latest.ckpt"))
            s += f"... save checkpoint to {os.path.join(save_dir, 'latest.ckpt')}"
            print(s)

        barrier()
        # -------- valid body -------- #
        if current_epoch % valid_interval == 0 or current_epoch == max_epochs - 1:

            s = time_log()
            s += f"| ***** Start valid epoch {current_epoch} / {max_epochs} (iter: {current_iter})"
            if is_master():
                print(s)

            valid_start_time = time.time()  # second
            valid_result = valid_epoch(model, valid_dataloader, train_cfg, device, current_iter)
            valid_time = time.time() - valid_start_time

            s = time_log()
            s += f"| ***** End valid epoch {current_epoch} / {max_epochs}, time: {valid_time:.3f}s\n"
            s += f"| ... acc: {valid_result['acc']:.4f}\n"
            s += f"| ... miou: {valid_result['miou']:.4f}\n"
            if is_master():
                print(s)

            current_miou = valid_result['miou']
            if current_best_miou <= current_miou:
                s = time_log()
                s += f"| ***** Best updated!\n" \
                     f"| ... previous best was at {best_epoch} epoch, {best_iter} iters\n" \
                     f"| ... acc: {current_best_miou:.4f} (prev) -> {current_miou:.4f} (new)\n"
                current_best_miou = current_miou
                best_iter = current_iter
                best_epoch = current_epoch

                if is_master():
                    # save checkpoint
                    ckpt = OrderedDict()
                    ckpt["model"] = model_m.state_dict()
                    ckpt["optimizer"] = optimizer.state_dict()
                    ckpt["scheduler"] = scheduler.state_dict()
                    ckpt["scaler"] = scaler.state_dict()
                    ckpt["stats"] = OrderedDict(epoch=current_epoch, iter=current_iter, max_epochs=max_epochs,
                                                best_acc=current_best_miou)
                    torch.save(ckpt, os.path.join(save_dir, "best.ckpt"))
                    s += f"| ... save checkpoint to {os.path.join(save_dir, 'best.ckpt')}"
                    print(s)
            else:
                s = time_log()
                s += f"| ***** Best not updated\n" \
                     f"| ... previous best was at {best_epoch} epoch, {best_iter} iters\n" \
                     f"| ... unseen_acc: {current_best_miou:.4f} (best) vs. {current_miou:.4f} (now)\n"
                if is_master():
                    print(s)

        barrier()
        scheduler.step()
        current_epoch += 1


if __name__ == '__main__':
    args, config = prepare_config()
    run(config, args.debug, args.eval)
