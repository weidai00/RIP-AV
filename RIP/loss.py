import torch.nn.functional as F
from torch import nn
import torch 
import numpy as np
from torch.autograd import Variable


arteryWeight = 3
veinWeight = 3
vesselWeight = 2



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
            labels_av_multi = torch.ones((indices.shape[0], 2)).to(targs.device)
            loss_av_multi = self.logitsLoss(selected_predictions_sigmoid, labels_av_multi)

            loss = (loss_a_v_av + loss_av_multi)
        else:
            loss = loss_a_v_av

        return loss

class multiLabelLossV2():
    def __init__(self):
        #self.ce = nn.CrossEntropyLoss()
        self.logitsLoss =  nn.BCEWithLogitsLoss()
    def __call__(self, preds, targs):
        #print(preds.shape, targs.shape)


        replacement_tensor = torch.FloatTensor([[1, 0], [0, 1], [1, 1]]).to(targs.device)
        new_tensor = replacement_tensor[targs]
        #print(new_tensor)
        loss = self.logitsLoss(preds,new_tensor)
  
        return loss,new_tensor


