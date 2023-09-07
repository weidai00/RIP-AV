import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# import natsort
import pandas as pd
from skimage import morphology
from sklearn import metrics
from Tools.BGR2RGB import BGR2RGB
from Tools.BinaryPostProcessing import binaryPostProcessing3
from PIL import Image

#########################################
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


#########################################

def AVclassifiationMetrics_skeletonPixles(PredAll1, PredAll2, VesselPredAll, LabelAll1, LabelAll2, LabelVesselAll,
                                          DataSet=0, onlyMeasureSkeleton=False, strict_mode=True):
    """
    predAll1: 动脉预测结果
    predAll2: 静脉预测结果
    VesselPredAll: 血管预测结果
    LabelAll1: 动脉标签
    LabelAll2: 静脉标签
    LabelVesselAll: 血管标签
    DataSet: 数据集大小
    onlyMeasureSkeleton: 是否只测量骨架
    strict_mode: 是否严格模式

    """

    ImgN = DataSet

    senList = []
    specList = []
    accList = []
    f1List = []
    ioulist = []
    diceList = []

    senList_sk = []
    specList_sk = []
    accList_sk = []
    f1List_sk = []
    ioulist_sk = []
    diceList_sk = []

    bad_case_count = 0
    bad_case_index = []
    if ImgN == 200:
        DF_disc = pd.read_excel('./Tools/DiskParameters_DRIVE_Test.xls', sheet_name=0)
    for ImgNumber in range(ImgN):

        height, width = PredAll1.shape[2:4]
        if ImgN == 200:
            discCenter = (DF_disc.loc[ImgNumber, 'DiskCenterRow'], DF_disc.loc[ImgNumber, 'DiskCenterCol'])
            discRadius = DF_disc.loc[ImgNumber, 'DiskRadius']
            MaskDisc = np.ones((height, width), np.uint8)
            cv2.circle(MaskDisc, center=(discCenter[1], discCenter[0]), radius=discRadius, color=0, thickness=-1)

        VesselProb = VesselPredAll[ImgNumber, 0, :, :]
        VesselLabel = LabelVesselAll[ImgNumber, 0, :, :]

        ArteryLabel = LabelAll1[ImgNumber, 0, :, :]
        VeinLabel = LabelAll2[ImgNumber, 0, :, :]

        ArteryProb = PredAll1[ImgNumber, 0, :, :]
        VeinProb = PredAll2[ImgNumber, 0, :, :]
        if ImgN == 200:
            VesselProb = cv2.bitwise_and(VesselProb, VesselProb, mask=MaskDisc)
            VesselLabel = cv2.bitwise_and(VesselLabel, VesselLabel, mask=MaskDisc)
            ArteryLabel = cv2.bitwise_and(ArteryLabel, ArteryLabel, mask=MaskDisc)
            VeinLabel = cv2.bitwise_and(VeinLabel, VeinLabel, mask=MaskDisc)
            ArteryProb = cv2.bitwise_and(ArteryProb, ArteryProb, mask=MaskDisc)
            VeinProb = cv2.bitwise_and(VeinProb, VeinProb, mask=MaskDisc)

        #########################################################
        """Only measure the AV classificaiton metrics on the segmented vessels, while the not segmented ones are not counted"""

        #        Artery = ArteryProb>=0.5
        #        Vein = VeinProb>=0.5
        #        VesselSeg = Artery + Vein
        if strict_mode:
            VesselSeg = VesselLabel
        else:
            VesselSeg = VesselProb >= 0.5
            VesselSeg = binaryPostProcessing3(VesselSeg, removeArea=100, fillArea=20)

        vesselPixels = np.where(VesselSeg > 0)

        ArteryProb2 = np.zeros((height, width))
        VeinProb2 = np.zeros((height, width))
        for i in range(len(vesselPixels[0])):
            row = vesselPixels[0][i]
            col = vesselPixels[1][i]
            probA = ArteryProb[row, col]
            probV = VeinProb[row, col]
            softmaxProb = softmax(np.array([probA, probV]))  # 是否可以改进
            ArteryProb2[row, col] = softmaxProb[0]
            VeinProb2[row, col] = softmaxProb[1]
            # ArteryProb2[row, col] = probA
            # VeinProb2[row, col] = probV

        ArteryLabelImg2 = ArteryLabel.copy()
        VeinLabelImg2 = VeinLabel.copy()
        ArteryLabelImg2[VesselSeg == 0] = 0
        VeinLabelImg2[VesselSeg == 0] = 0
        ArteryVeinLabelImg = np.zeros((height, width, 3), np.uint8)
        ArteryVeinLabelImg[ArteryLabelImg2 > 0] = (255, 0, 0)
        ArteryVeinLabelImg[VeinLabelImg2 > 0] = (0, 0, 255)
        ArteryVeinLabelCommon = np.bitwise_and(ArteryLabelImg2 > 0, VeinLabelImg2 > 0)

        ArteryPred2 = ArteryProb >= 0.5
        VeinPred2 = VeinProb >= 0.5

        ArteryPred2 = binaryPostProcessing3(ArteryPred2, removeArea=100, fillArea=20)
        VeinPred2 = binaryPostProcessing3(VeinPred2, removeArea=100, fillArea=20)
        ArteryVeinPredImg_v2 = np.zeros((height, width, 3), np.uint8)

        ArteryVeinPredImg_v2[ArteryPred2 > 0] = (255, 0, 0)
        ArteryVeinPredImg_v2[VeinPred2 > 0] = (0, 0, 255)
        ##################################################################################################
        """Get the ArteryVeinPredImg with Wrong Pixels Marked on the image"""
        ArteryVeinPredImg = np.zeros((height, width, 3), np.uint8)
        TPimg = np.bitwise_and(ArteryPred2 > 0, ArteryLabelImg2 > 0)  # 真实为动脉，预测为动脉
        TNimg = np.bitwise_and(VeinPred2 > 0, VeinLabelImg2 > 0)  # 真实为静脉，预测为静脉
        FPimg = np.bitwise_and(ArteryPred2 > 0, VeinLabelImg2 > 0)  # 真实为静脉，预测为动脉
        FPimg = np.bitwise_and(FPimg, np.bitwise_not(ArteryVeinLabelCommon))  # 真实为静脉，预测为动脉，且不属于动静脉共存区域

        FNimg = np.bitwise_and(VeinPred2 > 0, ArteryLabelImg2 > 0)  # 真实为动脉，预测为静脉
        FNimg = np.bitwise_and(FNimg, np.bitwise_not(ArteryVeinLabelCommon))  # 真实为动脉，预测为静脉，且不属于动静脉共存区域

        ArteryVeinPredImg[TPimg > 0, :] = (255, 0, 0)  # 动脉
        ArteryVeinPredImg[TNimg > 0, :] = (0, 0, 255)  # 静脉
        ArteryVeinPredImg[FPimg > 0, :] = (0, 255, 255)  # 错误预测为静脉
        ArteryVeinPredImg[FNimg > 0, :] = (255, 255, 0)  # 错误预测为动脉
        # Image.fromarray(ArteryVeinPredImg_v2).save(f'av_{ImgNumber}_with_dis_V2.png')
        ##################################################################################################
        """Calculate pixel-wise sensitivity, specificity and accuracy"""

        if not onlyMeasureSkeleton:
            TPa = np.count_nonzero(TPimg)
            TNa = np.count_nonzero(TNimg)
            FPa = np.count_nonzero(FPimg)
            FNa = np.count_nonzero(FNimg)

            sensitivity = TPa / (TPa + FNa)
            specificity = TNa / (TNa + FPa)
            acc = (TPa + TNa) / (TPa + TNa + FPa + FNa)
            f1 = 2 * TPa / (2 * TPa + FPa + FNa)
            dice = 2 * TPa / (2 * TPa + FPa + FNa)
            iou = TPa / (TPa + FPa + FNa)
            # print('Pixel-wise Metrics', acc, sensitivity, specificity)

            senList.append(sensitivity)
            specList.append(specificity)
            accList.append(acc)
            f1List.append(f1)
            diceList.append(dice)
            ioulist.append(iou)
        # print('Avg Per:', np.mean(accList), np.mean(senList), np.mean(specList))

        ##################################################################################################
        """Skeleton Performance Measurement"""
        Skeleton = np.uint8(morphology.skeletonize(VesselSeg))
        # np.save('./tmpfile/tmp_skeleton'+str(ImgNumber)+'.npy',Skeleton)

        ArterySkeletonLabel = cv2.bitwise_and(ArteryLabelImg2, ArteryLabelImg2, mask=Skeleton)
        VeinSkeletonLabel = cv2.bitwise_and(VeinLabelImg2, VeinLabelImg2, mask=Skeleton)

        ArterySkeletonPred = cv2.bitwise_and(ArteryPred2, ArteryPred2, mask=Skeleton)
        VeinSkeletonPred = cv2.bitwise_and(VeinPred2, VeinPred2, mask=Skeleton)

        ArteryVeinPred_sk = np.zeros((height, width, 3), np.uint8)
        skeletonPixles = np.where(Skeleton > 0)

        TPa_sk = 0
        TNa_sk = 0
        FPa_sk = 0
        FNa_sk = 0
        for i in range(len(skeletonPixles[0])):
            row = skeletonPixles[0][i]
            col = skeletonPixles[1][i]
            if ArterySkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1:
                TPa_sk = TPa_sk + 1
                ArteryVeinPred_sk[row, col] = (255, 0, 0)
            elif VeinSkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1:
                TNa_sk = TNa_sk + 1
                ArteryVeinPred_sk[row, col] = (0, 0, 255)
            elif ArterySkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1 \
                    and ArteryVeinLabelCommon[row, col] == 0:
                FNa_sk = FNa_sk + 1
                ArteryVeinPred_sk[row, col] = (255, 255, 0)
            elif VeinSkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1 \
                    and ArteryVeinLabelCommon[row, col] == 0:
                FPa_sk = FPa_sk + 1
                ArteryVeinPred_sk[row, col] = (0, 255, 255)
            else:
                pass
        # print("Img_index:"+str(ImgNumber))
        if (TPa_sk + FNa_sk) == 0 and (TNa_sk + FPa_sk) == 0 and (TPa_sk + TNa_sk + FPa_sk + FNa_sk) == 0:
            bad_case_count += 1
            bad_case_index.append(ImgNumber)
        sensitivity_sk = TPa_sk / (TPa_sk + FNa_sk)
        specificity_sk = TNa_sk / (TNa_sk + FPa_sk)
        acc_sk = (TPa_sk + TNa_sk) / (TPa_sk + TNa_sk + FPa_sk + FNa_sk)
        f1_sk = 2 * TPa_sk / (2 * TPa_sk + FPa_sk + FNa_sk)
        dice_sk = 2 * TPa_sk / (2 * TPa_sk + FPa_sk + FNa_sk)
        iou_sk = TPa_sk / (TPa_sk + FPa_sk + FNa_sk)

        senList_sk.append(sensitivity_sk)
        specList_sk.append(specificity_sk)
        accList_sk.append(acc_sk)
        f1List_sk.append(f1_sk)
        diceList_sk.append(dice_sk)
        ioulist_sk.append(iou_sk)
        # print('Skeletonal Metrics', acc_sk, sensitivity_sk, specificity_sk)
    if onlyMeasureSkeleton:
        print('Avg Skeleton Performance:', np.mean(accList_sk), np.mean(senList_sk), np.mean(specList_sk))
        return np.mean(accList_sk), np.mean(specList_sk), np.mean(senList_sk), np.mean(f1List_sk), np.mean(
            diceList_sk), np.mean(ioulist_sk), bad_case_index
    else:
        print('Avg Pixel-wise Performance:', np.mean(accList), np.mean(senList), np.mean(specList))
        return np.mean(accList), np.mean(specList), np.mean(senList), np.mean(f1List), np.mean(diceList), np.mean(
            ioulist)






if __name__ == '__main__':
    pro_path_post = r'G:\Tencent\WeChat\WeChat Files\wxid_3yy1snyq84jn21\FileStorage\File\2023-05\ProMap_testset_after_process_v2.npy'
    #pro_path = r'G:\Tencent\WeChat\WeChat Files\wxid_3yy1snyq84jn21\FileStorage\File\2023-05\ProMap_testset_original.npy'
    pro_path = r'E:\eye_paper\AUV-GAN\log\2023_05_27_17_50_09\running_result\ProMap_testset.npy'
    lable_path = r'E:\eye_paper\AUV-GAN\log\2023_05_27_17_50_09\running_result\LabelMap_testset.npy'
    mask_path = r'E:\eye_paper\AUV-GAN\log\2023_05_27_17_50_09\running_result\MaskAll_testset.npy'
    ps = np.load(pro_path)
    ps_pos = np.load(pro_path_post)
    ls = np.load(lable_path)
    mask = np.load(mask_path)

    AVclassifiationMetrics_skeletonPixles(ps[:, 0:1, :, :], ps[:, 1:2, :, :], ps[:, 2:, :, :], ls[:, 0:1, :, :],
                                          ls[:, 1:2, :, :], ls[:, 2:, :, :], DataSet=ps.shape[0],strict_mode=False)