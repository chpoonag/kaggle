# -*- coding: utf-8 -*-
"""
Metrics used to evaluate the outlier detection performance
"""
# Author: Yingtong Dou <ytongdou@gmail.com>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)

import torch

def eval_roc_auc(label, score):
    """
    ROC-AUC score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    roc_auc : float
        Average ROC-AUC score across different labels.
    """
    # WIP; encountered some error here; 20241104
    roc_auc = roc_auc_score(y_true=label, y_score=score)
    return roc_auc


def eval_recall_at_k(label, score, k=None):
    """
    Recall score for top k instances with the highest outlier scores.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.
    k : int, optional
        The number of instances to evaluate. ``None`` for
        recall. Default: ``None``.

    Returns
    -------
    recall_at_k : float
        Recall for top k instances with the highest outlier scores.
    """

    if k is None:
        k = sum(label)
    recall_at_k = sum(label[score.topk(k).indices]) / sum(label)
    return recall_at_k


def eval_precision_at_k(label, score, k=None):
    """
    Precision score for top k instances with the highest outlier scores.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.
    k : int, optional
        The number of instances to evaluate. ``None`` for
        precision. Default: ``None``.

    Returns
    -------
    precision_at_k : float
        Precision for top k instances with the highest outlier scores.
    """

    if k is None:
        k = sum(label)
    precision_at_k = sum(label[score.topk(k).indices]) / k
    return precision_at_k


def eval_average_precision(label, score):
    """
    Average precision score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    ap : float
        Average precision score.
    """

    ap = average_precision_score(y_true=label, y_score=score)
    return ap


def eval_f1(label, pred):
    """
    F1 score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : torch.Tensor
        Outlier prediction in shape of ``(N, )``.

    Returns
    -------
    f1 : float
        F1 score.
    """

    f1 = f1_score(y_true=label, y_pred=pred)
    return f1


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()

class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=2):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size
        self._eps = 1e-6
        
    def reset(self):
        self.matrix = torch.zeros(self.size, self.size)

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

#     @property
#     def class_iou(self):
#         true_pos = self.matrix.diagonal()
#         return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + self._eps)

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        denom = self.matrix.sum(0) + self.matrix.sum(1) - true_pos
        denom = torch.max(denom, torch.tensor(self._eps))
        return true_pos / denom

    @property
    def iou(self):
        return self.class_iou.mean()

#     @property
#     def global_accuracy(self):
#         true_pos = self.matrix.diagonal()
#         return true_pos.sum() / (self.matrix.sum() + self._eps)

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        denom = self.matrix.sum()
        denom = torch.max(denom, torch.tensor(self._eps))
        return true_pos.sum() / denom

#     @property
#     def class_accuracy(self):
#         true_pos = self.matrix.diagonal()
#         return true_pos / (self.matrix.sum(1) + self._eps)
    
    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        denom = self.matrix.sum(1)
        denom = torch.max(denom, torch.tensor(self._eps))
        return true_pos / denom

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + self._eps)
    
#     @property
#     def precision(self):
#         true_pos = self.matrix.diagonal()
#         return true_pos / (self.matrix.sum(0) + self._eps)

#     @property
#     def recall(self):
#         true_pos = self.matrix.diagonal()
#         return true_pos / (self.matrix.sum(1) + self._eps)

#     @property
#     def f1_score(self):
#         precision = self.precision
#         recall = self.recall
#         return 2 * (precision * recall) / (precision + recall + self._eps)

    @property
    def precision(self):
        true_pos = self.matrix.diagonal()
        denom = self.matrix.sum(0)
        denom = torch.max(denom, torch.tensor(self._eps))
        return true_pos / denom

    @property
    def recall(self):
        true_pos = self.matrix.diagonal()
        denom = self.matrix.sum(1)
        denom = torch.max(denom, torch.tensor(self._eps))
        return true_pos / denom

    @property
    def f1_score(self):
        precision = self.precision
        recall = self.recall
        denom = precision + recall
        denom = torch.max(denom, torch.tensor(self._eps))
        return 2 * (precision * recall) / denom
    
    def get_report(self, class_names=None, digits=4, print_report=True):
        """
        Prints a classification report similar to sklearn's classification_report.
        :param class_names: A list of class names (optional)
        :param digits: Number of digits for formatting output floating point values (default=2)
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.size)]

        # Compute support (number of occurrences of each class)
        support = self.matrix.sum(1)

        # Headers
        headers = ["precision", "recall", "f1-score", "support"]
        name_width = max(len(cn) for cn in class_names)
        width = max(name_width, 8, len("average"))
        head_fmt = "{:>{width}} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"

        # Rows
        row_fmt = "{:>{width}} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
        for i, class_name in enumerate(class_names):
            report += row_fmt.format(
                class_name,
                self.precision[i].item(),
                self.recall[i].item(),
                self.f1_score[i].item(),
                support[i].int().item(),
                width=width,
                digits=digits
            )
           
        report += f"Accuracy:\t{self.global_accuracy:.{digits}f}"
        if print_report:
            print(report)
        return report