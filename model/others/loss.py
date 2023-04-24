import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class bootstrap_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_list = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    def forward(self, masks: torch.Tensor,
                pseudo_label: torch.Tensor,
                pseudo_label_teacher: torch.Tensor,
                epoch: int):
        '''

        :param masks: (b, n_cls, hw, [0, 1], fg+bg)
        :param pseudo_label: (b, hw) binary
        :param pseudo_label_teacher: (b, hw) binary
        '''

        N, N_cls_fgbg, h, w = masks.shape

        pseudo_label_blur = pseudo_label
        pseudo_label_norm = torch.stack(
            [(label - label.min()) / (label.max() - label.min()) for label in pseudo_label_blur])

        if pseudo_label_teacher is not None:
            w_1 = 0.5
            w_2 = 0.5
            pseudo_label_teacher_blur = pseudo_label_teacher
            pseudo_label_teacher_norm = pseudo_label_teacher_blur.softmax(1)
            pseudo_label_norm = w_1 * pseudo_label_teacher_norm + w_2 * pseudo_label_norm

        bootstrapped_pseudo_labels = pseudo_label_norm.max(1)[1]

        if pseudo_label_teacher is not None:

            alpha = self.alpha_list[-1] if not (epoch < len(self.alpha_list)) else self.alpha_list[epoch]

            # negative sample: random shuffle
            idx_list = list(range(N_cls_fgbg))
            random.shuffle(idx_list)
            bootstrapped_pseudo_labels_negative = pseudo_label_norm[:, idx_list].max(1)[1]

            loss_cat = F.cross_entropy(masks, bootstrapped_pseudo_labels, reduce=False) \
                       - alpha * F.cross_entropy(masks, bootstrapped_pseudo_labels_negative, reduce=False)
        else:
            loss_cat = F.cross_entropy(masks, bootstrapped_pseudo_labels, reduce=False)

        mask_topk = masks.view(N, N_cls_fgbg, h, w).softmax(1).topk(N_cls_fgbg, dim=1)[0]
        loss_uncertainty = 1 - (mask_topk[:, 0] - mask_topk[:, 1])

        return loss_cat, loss_uncertainty, bootstrapped_pseudo_labels


class cls_emb_loss(nn.Module):
    def __init__(self, dist="cos"):
        super().__init__()
        if dist == 'cos':
            pass
        else:
            raise NotImplementedError

    def forward(self, cls_emb: torch.Tensor):
        '''

        :param cls_emb: (b, n_cls, c, [0, 1], normalized)
        :return:
        '''
        N, N_cls, N_dim = cls_emb.shape
        dist = cls_emb @ cls_emb.transpose(1, 2)
        dist = dist.triu(1)
        dist_ = dist.masked_select(torch.ones_like(dist).triu(1).bool()).view(N, -1)
        loss = 1 + dist_

        return loss
