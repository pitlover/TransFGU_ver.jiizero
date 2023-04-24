import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)  # Exclude unlabelled data.
    hist = np.bincount(n_class * label_true[mask] + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist


def UnsegMetrics(histogram, n_cls, prefix=None):
    m = linear_assignment(histogram.max() - histogram)

    hist = np.zeros((n_cls, n_cls))

    for idx in range(n_cls):
        hist[m[1][idx]] = histogram[idx]

    tp = np.diag(hist)
    fp = np.sum(hist, 0) - tp
    fn = np.sum(hist, 1) - tp

    iou = tp / (tp + fp + fn)
    prc = tp / (tp + fn)
    opc = np.sum(tp) / np.sum(hist)

    result = {"iou": iou,
              "mean_iou": np.nanmean(iou),
              "precision_per_class (per class accuracy)": prc,
              "mean_precision (class-avg accuracy)": np.nanmean(prc),
              "overall_precision (pixel accuracy)": opc}

    result = {k: 100 * v for k, v in result.items()}

    return result


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
