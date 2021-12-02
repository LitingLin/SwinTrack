# https://github.com/hyz-xmaster/VarifocalNet/blob/master/mmdet/models/losses/varifocal_loss.py
import torch.nn as nn
import torch.nn.functional as F


def varifocal_loss(pred,
                   target,
                   alpha=0.75,
                   gamma=2.0,
                   iou_weighted=True,
                   use_sigmoid=True):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        use_sigmoid (bool, optional): Whether the prediction is
            used for sigmoid or softmax. Defaults to True.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    if use_sigmoid:
        pred_sigmoid = pred.sigmoid()
    else:
        pred_sigmoid = pred
    target = target.type_as(pred)
    if iou_weighted:
        focal_weight = target * (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    else:
        focal_weight = (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    loss = F.binary_cross_entropy(
        pred_sigmoid, target, reduction='none') * focal_weight
    return loss


class VarifocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.75,
                 gamma=2.0,
                 iou_weighted=True):
        """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
        Args:
            use_sigmoid (bool, optional): Whether the prediction is
                used for sigmoid or softmax. Defaults to True.
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal
                Loss. Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive examples with the iou target. Defaults to True.
        """
        super(VarifocalLoss, self).__init__()
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted

    def forward(self,
                pred,
                target):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
        Returns:
            torch.Tensor: The calculated loss
        """
        return varifocal_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            iou_weighted=self.iou_weighted,
            use_sigmoid=self.use_sigmoid)
