from criterion.common.reduction.default import build_loss_reduction_function
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(pred, target, use_sigmoid=True):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    # label denotes the category id, score denotes the quality score
    # N, L
    if use_sigmoid:
        func = F.binary_cross_entropy_with_logits
    else:
        func = F.binary_cross_entropy

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid() if use_sigmoid else pred

    loss = func(pred_sigmoid, target, reduction='none')
    return loss.flatten()


class BCELoss(nn.Module):
    """
    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
    """

    def __init__(self,
                 use_sigmoid=True):
        super(BCELoss, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid

    def forward(self,
                pred,
                target):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
        """
        return bce_loss(
            pred,
            target,
            use_sigmoid=self.use_sigmoid)


def cls_data_adaptor(predicted, label, _):
    cls_score = predicted['class_score']
    cls_score = cls_score.flatten(2).transpose(1, 2).flatten(1)
    return True, (cls_score, label['class_label'])


def build_binary_cross_entropy(loss_parameters, *_):
    return BCELoss(False), cls_data_adaptor, build_loss_reduction_function(loss_parameters)
