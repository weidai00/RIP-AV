import torch.nn.functional as F
from torch import nn
import torch 
import numpy as np
from torch.autograd import Variable
from scipy.ndimage.morphology import distance_transform_edt as edt

arteryWeight = 3
veinWeight = 3
vesselWeight = 2

def centernessLoss(criterion, centerness_maps, label_centerness_map, v1, weight=1, epsilon=1e-12):
    #
    smoothl1 = criterion(centerness_maps, label_centerness_map)

    square_centerness_map = label_centerness_map + epsilon
    # square_centerness_map = label_centerness_map * label_centerness_map + epsilon

    term1 = smoothl1 / square_centerness_map

    bs, ch, h, w = centerness_maps.shape

    term1_0 = term1[:,0:3,:,:]

    term1_sum0 = torch.sum(term1_0)

    loss = term1_sum0 / v1 * weight



    return loss
class multiclassLoss():
    def __init__(self,num_classes=3):
        self.num_classes = num_classes

        # self.logitsLoss = nn.BCEWithLogitsLoss()
        self.logitsLoss = nn.BCELoss()
    def __call__(self, preds, targs):

        target_artery = targs[:,0,:,:]
        target_vein = targs[:,1,:,:]
        target_all = targs[:,2,:,:]


        loss = ( arteryWeight*self.logitsLoss(preds[:,0], target_artery) +
                 veinWeight*self.logitsLoss(preds[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(preds[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight)


        return loss

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        #predict = torch.sigmoid(predict)
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2*torch.sum(torch.mul(predict, target)) + self.smooth
        #den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth
        den = torch.sum(predict) + torch.sum(target) + self.smooth
        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class multidiceLoss():
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        # self.logitsLoss = nn.BCEWithLogitsLoss()
        self.logitsdiceLoss = BinaryDiceLoss()

    def __call__(self, preds, targs):
        # print(preds.shape, targs.shape)

        #        target_artery = (targs == 2).float()
        #        target_vein = (targs == 1).float()
        #        target_all = (targs >= 1).float()
        target_artery = targs[:, 0, :, :]
        target_vein = targs[:, 1, :, :]
        target_all = targs[:, 2, :, :]


        loss = (arteryWeight * self.logitsdiceLoss(preds[:, 0], target_artery) +
                     veinWeight * self.logitsdiceLoss(preds[:, 2], target_vein) +
                     vesselWeight * self.logitsdiceLoss(preds[:, 1], target_all)) / (
                                arteryWeight + veinWeight + vesselWeight)



        return loss


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform
    Hausdorff loss implementation based on paper:
    https://arxiv.org/pdf/1904.10030.pdf

    copy pasted from - all credit goes to original authors:
    https://github.com/SilmarilBearer/HausdorffLoss
    """

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
            self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
                pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.detach().cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.detach().cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        #dt_field = pred_error * distance.cuda()
        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


class multiHausdorffDTLoss():
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        # self.logitsLoss = nn.BCEWithLogitsLoss()
        self.logitsHausdorffDTLoss = HausdorffDTLoss()

    def __call__(self, preds, targs):

        """
        preds: (b, 3, x, y, z) or (b, 3, x, y)
        targs: (b, 3, x, y, z) or (b, 3, x, y)
        """

        target_artery = targs[:, 0:1, :, :]
        target_vein = targs[:, 1:2, :, :]
        target_all = targs[:, 2:, :, :]
        loss = (arteryWeight * self.logitsHausdorffDTLoss(preds[:, 0:1], target_artery) +
                veinWeight * self.logitsHausdorffDTLoss(preds[:, 2:], target_vein) +
                vesselWeight * self.logitsHausdorffDTLoss(preds[:, 1:2], target_all)) / (
                       arteryWeight + veinWeight + vesselWeight)

        return loss

class multiLabelLoss():
    def __init__(self):
        self.ce = nn.CrossEntropyLoss()
        self.logitsLoss = nn.BCELoss()
    def __call__(self, preds, targs):
        #print(preds.shape, targs.shape)

        loss_a_v_av = self.ce(preds, targs)

        label_av = targs.clone()
        indices = torch.nonzero(label_av == 2).squeeze(1)
        selected_predictions = []

        # 根据索引提取对应的预测样本
        if indices.shape[0] != 0:
            selected_predictions = preds[indices][:, :2]
            selected_predictions_sigmoid = torch.sigmoid(selected_predictions)
            labels_av_multi = torch.ones((indices.shape[0], 2))
            loss_av_multi = self.logitsLoss(selected_predictions_sigmoid, labels_av_multi)

            loss = loss_a_v_av + loss_av_multi/2
        else:
            loss = loss_a_v_av

        return loss

class multiclassLossAV():
    def __init__(self, num_classes=3):
        self.num_classes = num_classes

        # self.logitsLoss = nn.BCEWithLogitsLoss()
        self.logitsLoss = nn.BCELoss()

    def __call__(self, preds, targs):
        # print(preds.shape, targs.shape)

        #        target_artery = (targs == 2).float()
        #        target_vein = (targs == 1).float()
        #        target_all = (targs >= 1).float()
        target_artery = targs[:, 0, :, :]
        target_vein = targs[:, 1, :, :]
        target_all = targs[:, 2, :, :]

        loss = (arteryWeight * self.logitsLoss(preds[:, 0], target_artery) +
                veinWeight * self.logitsLoss(preds[:, 2], target_vein) +
                vesselWeight * self.logitsLoss(preds[:, 1], target_all)) / (arteryWeight + veinWeight + vesselWeight)


        return loss