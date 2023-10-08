import numpy as np


class StreamSegMetrics:
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        hist = self.confusion_matrix + 10e-6

        tp = np.zeros(self.n_classes, dtype=np.float32)
        tn = np.zeros(self.n_classes, dtype=np.float32)
        cls_acc = np.zeros(self.n_classes, dtype=np.float32)
        cls_precision = np.zeros(self.n_classes, dtype=np.float32)
        cls_recall = np.zeros(self.n_classes, dtype=np.float32)
        cls_specificity = np.zeros(self.n_classes, dtype=np.float32)
        cls_f1 = np.zeros(self.n_classes, dtype=np.float32)
        for i in range(self.n_classes):
            tp[i] = hist[i, i]
            x = np.delete(hist, [i, i], axis=0)
            x = np.delete(x, [i, i], axis=1)
            tn[i] = x.sum()
            cls_acc[i] = (tp[i] + tn[i]) / hist.sum()
            cls_precision[i] = tp[i] / hist[:, i].sum()
            cls_recall[i] = tp[i] / hist[i].sum()
            cls_specificity[i] = tn[i] / np.delete(hist, [i, i], axis=0).sum()
            cls_f1[i] = 2 * cls_precision[i] * cls_recall[i] / (cls_precision[i] + cls_recall[i])

        acc = cls_acc.mean()
        precision = cls_precision.mean()
        recall = cls_recall.mean()
        specificity = cls_specificity.mean()
        f1 = cls_f1.mean()

        cls_acc = dict(zip(range(self.n_classes), cls_acc))
        cls_precision = dict(zip(range(self.n_classes), cls_precision))
        cls_specificity = dict(zip(range(self.n_classes), cls_specificity))
        cls_recall = dict(zip(range(self.n_classes), cls_recall))
        cls_f1 = dict(zip(range(self.n_classes), cls_f1))

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "acc": acc,
            "cls_acc": cls_acc,
            "m_precision": precision,
            "cls_precision": cls_precision,
            "m_recall": recall,
            "cls_recall": cls_recall,
            "m_specificity": specificity,
            "cls_specificity": cls_specificity,
            "m_f1": f1,
            "cls_f1": cls_f1,
            "m_IoU": mean_iu,
            "cls_IoU": cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
