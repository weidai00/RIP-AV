import torch.nn.functional as F
from torch import nn
import torch 
import numpy as np
from torch.autograd import Variable
from scipy.ndimage.morphology import distance_transform_edt as edt

arteryWeight = 3
veinWeight = 3
vesselWeight = 2


class SmoothL1_weighted():
    def __init__(self, weight_list=[]):
        self.smoothL1Loss = nn.SmoothL1Loss(reduction='none')
        self.weight_list = weight_list

    def __call__(self, preds, targs):
        bs, ch, h, w = preds.shape
        term = self.smoothL1Loss(preds, targs)
        v = bs * 3 * h * w
        term0 = torch.sum(term[:,0:3,:,:]) / v
        term1 = torch.sum(term[:,3:6,:,:]) / v if ch >= 6 else None
        term2 = torch.sum(term[:,6:9,:,:]) / v if ch >= 9 else None

        loss = term0 * self.weight_list[0]
        if ch >= 6:
            loss += term1 * self.weight_list[1]
        if ch >= 9:
            loss += term2 * self.weight_list[2]
        
        return loss
def centernessLoss(criterion, centerness_maps, label_centerness_map, v1, weight=1, epsilon=1e-12):
    # 2. calculate smooth l1
    smoothl1 = criterion(centerness_maps, label_centerness_map)
    # print("smoothl1:",smoothl1.shape, smoothl1.max(),smoothl1.min())
    # 3. calculate the square of predicted centerness map
    square_centerness_map = label_centerness_map + epsilon
    # square_centerness_map = label_centerness_map * label_centerness_map + epsilon
    # print("square_centerness_map:", square_centerness_map.shape, square_centerness_map.max(), square_centerness_map.min())
    # 4. calculate the 1/S^2 * smoothL1(S,S')
    term1 = smoothl1 / square_centerness_map

    bs, ch, h, w = centerness_maps.shape

    term1_0 = term1[:,0:3,:,:]

    term1_sum0 = torch.sum(term1_0)


    # 6. loss
    loss  = 0
    loss1 = term1_sum0 / v1 * weight

    loss = loss1

    return loss

def tripletMarginLoss_vggfea(vggnet, criterion, preds, targs, use_cuda=False,  weight_list=[1,1,1]):
    loss = 0
    # artery
    feat_pred_a = vggnet(torch.cat([preds[:, 0:1], preds[:, 0:1], preds[:, 0:1]], dim=1))
    feat_label_a = vggnet(torch.cat([targs[:, 0:1], targs[:, 0:1], targs[:, 0:1]], dim=1))

    # vein
    feat_pred_v = vggnet(torch.cat([preds[:, 2:], preds[:, 2:], preds[:, 2:]], dim=1))
    feat_label_v = vggnet(torch.cat([targs[:, 1:2], targs[:, 1:2], targs[:, 1:2]], dim=1))

    # vessel
    feat_pred_ves = vggnet(torch.cat([preds[:, 1:2], preds[:, 1:2], preds[:, 1:2]], dim=1))
    feat_label_ves = vggnet(torch.cat([targs[:, 2:], targs[:, 2:], targs[:, 2:]], dim=1))

    N = len(feat_pred_a)
    for i in range(N):
        loss += criterion(feat_pred_a[i], feat_label_a[i]) * weight_list[0] + \
           criterion(feat_pred_v[i], feat_label_v[i]) * weight_list[1] + \
           criterion(feat_pred_ves[i], feat_label_ves[i]) * weight_list[2]

    return loss

def vggloss(vggnet, criterion, preds, targs, use_cuda=False,  weight_list=[1,1,1]):
    loss = 0
    # artery
    feat_pred_a = vggnet(torch.cat([preds[:, 0:1], preds[:, 0:1], preds[:, 0:1]], dim=1))
    feat_label_a = vggnet(torch.cat([targs[:, 0:1], targs[:, 0:1], targs[:, 0:1]], dim=1))
    # vein
    feat_pred_v = vggnet(torch.cat([preds[:, 2:], preds[:, 2:], preds[:, 2:]], dim=1))
    feat_label_v = vggnet(torch.cat([targs[:, 1:2], targs[:, 1:2], targs[:, 1:2]], dim=1))
    # vessel
    feat_pred_ves = vggnet(torch.cat([preds[:, 1:2], preds[:, 1:2], preds[:, 1:2]], dim=1))
    feat_label_ves = vggnet(torch.cat([targs[:, 2:], targs[:, 2:], targs[:, 2:]], dim=1))

    N = len(feat_pred_a)
    for i in range(N):
        loss += criterion(feat_pred_a[i], feat_label_a[i])*weight_list[0] + \
           criterion(feat_pred_v[i], feat_label_v[i])*weight_list[1] + \
           criterion(feat_pred_ves[i], feat_label_ves[i])*weight_list[2]
    return loss

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)

    dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

class CrossEntropyLossWithSmooth():
    def __init__(self, smooth, num_classes, use_cuda=False):
        self.nlloss = nn.KLDivLoss()
        self.logSoftmax= nn.LogSoftmax()
        self.smooth = smooth
        self.num_classes = num_classes
        self.use_cuda = use_cuda
        self.confidence = 1.0 - self.smooth
    def __call__(self, preds, targs):
        assert preds.size(1) == self.num_classes
        smooth_label = torch.ones_like(preds)
        if self.use_cuda:
            smooth_label = smooth_label.cuda()
        smooth_label.fill_(self.smooth / (self.num_classes - 1))
        smooth_label.scatter_(1, targs.data.unsqueeze(1), self.confidence)
        smooth_label = Variable(smooth_label, requires_grad=False)
        loss = self.nlloss(self.logSoftmax(preds), smooth_label)
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



class L1LossWithLogits():
    def __init__(self):
        self.l1loss = nn.L1Loss()

    def __call__(self, preds, targs):
        preds = torch.sigmoid(preds)
        target_artery = targs[:,0,:,:]
        target_vein = targs[:,1,:,:]
        target_all = targs[:,2,:,:]
        loss = self.l1loss(preds[:,0], target_artery) + \
               self.l1loss(preds[:,2], target_vein) + \
               self.l1loss(preds[:,1], target_all)
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

class multiclassLoss():
    def __init__(self,num_classes=3):
        self.num_classes = num_classes

        # self.logitsLoss = nn.BCEWithLogitsLoss()
        self.logitsLoss = nn.BCELoss()
        self.focalLoss_a = FocalLoss(alpha=0.1, gamma=2)
        self.focalLoss_v = FocalLoss(alpha=0.1, gamma=2)
        self.focalLoss_vessel = FocalLoss(alpha=0.2, gamma=2)
    def __call__(self, preds, targs):
        #print(preds.shape, targs.shape)
        
#        target_artery = (targs == 2).float()
#        target_vein = (targs == 1).float()
#        target_all = (targs >= 1).float()
        target_artery = targs[:,0,:,:]
        target_vein = targs[:,1,:,:]
        target_all = targs[:,2,:,:]


        # a_ori = preds[:,0:1].clone()
        # v_ori = preds[:,2:3].clone()
        # vessel = preds[:,1:2].clone()
        # predav = torch.cat((a_ori, v_ori), dim=1)
        # predav = F.softmax(predav, dim=1)
        # a = predav[:,0:1].clone()
        # v = predav[:,1:2].clone()
        # replaced = vessel>=0.5
        # a_ori[replaced] = a[replaced]
        # v_ori[replaced] = v[replaced]
        # preds = torch.cat((a_ori, vessel, v_ori), dim=1)

        loss = ( arteryWeight*self.logitsLoss(preds[:,0], target_artery) +
                 veinWeight*self.logitsLoss(preds[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(preds[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight)



        loss2 = (
            arteryWeight*self.focalLoss_a(preds[:,0], target_artery) +
            veinWeight*self.focalLoss_v(preds[:,2], target_vein) +
            vesselWeight*self.focalLoss_vessel(preds[:,1], target_all)/(arteryWeight+veinWeight+vesselWeight)


        )

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
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


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