import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# import natsort
import pandas as pd
from skimage import morphology
from sklearn import metrics
from Tools.BinaryPostProcessing import binaryPostProcessing3
from PIL import Image
from scipy.signal import convolve2d
import time

#########################################
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def Skeleton(a_or_v, a_and_v):
    th = np.uint8(a_and_v)
    # Distance transform for maximum diameter
    vessels = th.copy()
    dist = cv2.distanceTransform(a_or_v, cv2.DIST_L2, 3)  
    thinned = np.uint8(morphology.skeletonize((vessels / 255))) * 255
    return thinned, dist


def cal_crosspoint(vessel):
    # Removing bifurcation points by using specially designed kernels
    # Can be optimized further! (not the best implementation)
    thinned1, dist = Skeleton(vessel, vessel)
    thh = thinned1.copy()
    thh = thh / 255
    kernel1 = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    th = convolve2d(thh, kernel1, mode="same")
    for u in range(th.shape[0]):
        for j in range(th.shape[1]):
            if th[u, j] >= 13.0:
                cv2.circle(vessel, (j, u), 3 * int(dist[u, j]), (0, 0, 0), -1)
    # thi = cv2.cvtColor(thi, cv2.COLOR_BGR2GRAY)
    return vessel



def AVclassifiation_pos_ve(out_path, PredAll1, PredAll2, VesselPredAll, DataSet=0, image_basename=''):
    """
    predAll1: predition results of artery
    predAll2: predition results of vein
    VesselPredAll: predition results of vessel
    DataSet: the length of dataset
    image_basename: the name of saved mask
    """

    ImgN = DataSet

    for ImgNumber in range(ImgN):

        height, width = PredAll1.shape[2:4]

        VesselProb = VesselPredAll[ImgNumber, 0, :, :]

        ArteryProb = PredAll1[ImgNumber, 0, :, :]
        VeinProb = PredAll2[ImgNumber, 0, :, :]

        VesselSeg = (VesselProb >= 0.1) & ((ArteryProb >= 0.2) | (VeinProb >= 0.2))
        # VesselSeg = (VesselProb >= 0.5) & ((ArteryProb >= 0.5) | (VeinProb >= 0.5))
        crossSeg = (VesselProb >= 0.1) & ((ArteryProb >= 0.6) & (VeinProb >= 0.6))
        VesselSeg = binaryPostProcessing3(VesselSeg, removeArea=100, fillArea=20)

        vesselPixels = np.where(VesselSeg > 0)

        ArteryProb2 = np.zeros((height, width))
        VeinProb2 = np.zeros((height, width))
        crossProb2 = np.zeros((height, width))
        image_color = np.zeros((3, height, width), dtype=np.uint8)
        for i in range(len(vesselPixels[0])):
            row = vesselPixels[0][i]
            col = vesselPixels[1][i]
            probA = ArteryProb[row, col]
            probV = VeinProb[row, col]
            ArteryProb2[row, col] = probA
            VeinProb2[row, col] = probV

        test_use_vessel = np.zeros((height, width), np.uint8)
        ArteryPred2 = ((ArteryProb2 >= 0.2) & (ArteryProb2 > VeinProb2))
        VeinPred2 = ((VeinProb2 >= 0.2) & (VeinProb2 > ArteryProb2))

        ArteryPred2 = binaryPostProcessing3(ArteryPred2, removeArea=100, fillArea=20)
        VeinPred2 = binaryPostProcessing3(VeinPred2, removeArea=100, fillArea=20)

        image_color[0, :, :] = ArteryPred2 * 255
        image_color[2, :, :] = VeinPred2 * 255
        image_color = image_color.transpose((1, 2, 0))

        imgBin_vessel = ArteryPred2 + VeinPred2
        imgBin_vessel[imgBin_vessel[:, :] == 2] = 1
        test_use_vessel = imgBin_vessel.copy() * 255

        vessel = cal_crosspoint(test_use_vessel)

        contours_vessel, hierarchy_c = cv2.findContours(vessel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # inter continuity
        for vessel_seg in range(len(contours_vessel)):
            C_vessel = np.zeros(vessel.shape, np.uint8)
            C_vessel = cv2.drawContours(C_vessel, contours_vessel, vessel_seg, (255, 255, 255), cv2.FILLED)
            cli = np.mean(VeinProb2[C_vessel == 255]) / np.mean(ArteryProb2[C_vessel == 255])
            if cli < 1:
                image_color[
                    (C_vessel[:, :] == 255) & (test_use_vessel[:, :] == 255)] = [255, 0, 0]
            else:
                image_color[
                    (C_vessel[:, :] == 255) & (test_use_vessel[:, :] == 255)] = [0, 0, 255]


        Image.fromarray(image_color).save(os.path.join(out_path, f'{image_basename[ImgNumber].split(".")[0]}.png'))


def AVclassifiation(out_path, PredAll1, PredAll2, VesselPredAll, DataSet=0, image_basename=''):
    """
    predAll1: predition results of artery
    predAll2: predition results of vein
    VesselPredAll: predition results of vessel
    DataSet: the length of dataset
    image_basename: the name of saved mask
    """

    ImgN = DataSet

    for ImgNumber in range(ImgN):

        height, width = PredAll1.shape[2:4]

        VesselProb = VesselPredAll[ImgNumber, 0, :, :]

        ArteryProb = PredAll1[ImgNumber, 0, :, :]
        VeinProb = PredAll2[ImgNumber, 0, :, :]

        VesselSeg = (VesselProb >= 0.1) & ((ArteryProb >0.2) | (VeinProb > 0.2))
        # VesselSeg = (VesselProb >= 0.5) & ((ArteryProb >= 0.5) | (VeinProb >= 0.5))
        crossSeg = (VesselProb >= 0.1) & ((ArteryProb >= 0.6) & (VeinProb >= 0.6))
        VesselSeg = binaryPostProcessing3(VesselSeg, removeArea=100, fillArea=20)

        vesselPixels = np.where(VesselSeg > 0)

        ArteryProb2 = np.zeros((height, width))
        VeinProb2 = np.zeros((height, width))
        crossProb2 = np.zeros((height, width))
        image_color = np.zeros((3, height, width), dtype=np.uint8)
        for i in range(len(vesselPixels[0])):
            row = vesselPixels[0][i]
            col = vesselPixels[1][i]
            probA = ArteryProb[row, col]
            probV = VeinProb[row, col]
            #probA,probV = softmax([probA,probV])
            ArteryProb2[row, col] = probA
            VeinProb2[row, col] = probV

        test_use_vessel = np.zeros((height, width), np.uint8)
        ArteryPred2 = ((ArteryProb2 >= 0.2) & (ArteryProb2 >= VeinProb2))
        VeinPred2 = ((VeinProb2 >= 0.2) & (VeinProb2 >= ArteryProb2))

        ArteryPred2 = binaryPostProcessing3(ArteryPred2, removeArea=100, fillArea=20)
        VeinPred2 = binaryPostProcessing3(VeinPred2, removeArea=100, fillArea=20)

        image_color[0, :, :] = ArteryPred2 * 255
        image_color[2, :, :] = VeinPred2 * 255
        image_color = image_color.transpose((1, 2, 0))

        #Image.fromarray(image_color).save(os.path.join(out_path, f'{image_basename[ImgNumber].split(".")[0]}_ori.png'))

        imgBin_vessel = ArteryPred2 + VeinPred2
        imgBin_vessel[imgBin_vessel[:, :] == 2] = 1
        test_use_vessel = imgBin_vessel.copy() * 255

        vessel = cal_crosspoint(test_use_vessel)

        contours_vessel, hierarchy_c = cv2.findContours(vessel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # inter continuity
        for vessel_seg in range(len(contours_vessel)):
            C_vessel = np.zeros(vessel.shape, np.uint8)
            C_vessel = cv2.drawContours(C_vessel, contours_vessel, vessel_seg, (255, 255, 255), cv2.FILLED)
            cli = np.mean(VeinProb2[C_vessel == 255]) / np.mean(ArteryProb2[C_vessel == 255])
            if cli < 1:
                image_color[
                    (C_vessel[:, :] == 255) & (test_use_vessel[:, :] == 255)] = [255, 0, 0]
            else:
                image_color[
                    (C_vessel[:, :] == 255) & (test_use_vessel[:, :] == 255)] = [0, 0, 255]
        loop=0
        while loop<2:
            # out vein continuity
            vein = image_color[:, :, 2]
            contours_vein, hierarchy_b = cv2.findContours(vein, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            vein_size = []
            for z in range(len(contours_vein)):
                vein_size.append(contours_vein[z].size)
            vein_size = np.sort(np.array(vein_size))
            # image_color_copy = np.uint8(image_color).copy()
            for vein_seg in range(len(contours_vein)):
                judge_number = min(np.mean(vein_size),500)
                # cv2.putText(image_color_copy, str(vein_seg), (int(contours_vein[vein_seg][0][0][0]), int(contours_vein[vein_seg][0][0][1])), 3, 1,
                #             color=(255, 0, 0), thickness=2)
                if contours_vein[vein_seg].size < judge_number:
                    C_vein = np.zeros(vessel.shape, np.uint8)
                    C_vein = cv2.drawContours(C_vein, contours_vein, vein_seg, (255, 255, 255), cv2.FILLED)
                    max_diameter = np.max(Skeleton(C_vein, C_vein)[1])

                    image_color_copy_vein = image_color[:, :, 2].copy()
                    image_color_copy_arter = image_color[:, :, 0].copy()
                    # a_ori = cv2.drawContours(a_ori, contours_b, k, (0, 0, 0), cv2.FILLED)
                    image_color_copy_vein = cv2.drawContours(image_color_copy_vein, contours_vein, vein_seg,
                                                             (0, 0, 0),
                                                             cv2.FILLED)
                    # image_color[(C_cross[:, :] == 255) & (image_color[:, :, 1] == 255)] = [255, 0, 0]
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
                        4 * int(np.ceil(max_diameter)), 4 * int(np.ceil(max_diameter))))
                    C_vein_dilate = cv2.dilate(C_vein, kernel, iterations=1)
                    # cv2.imwrite(path_out_3, C_vein_dilate)
                    C_vein_dilate_judge = np.zeros(vessel.shape, np.uint8)
                    C_vein_dilate_judge[
                        (C_vein_dilate[:, :] == 255) & (image_color_copy_vein == 255)] = 1
                    C_arter_dilate_judge = np.zeros(vessel.shape, np.uint8)
                    C_arter_dilate_judge[
                        (C_vein_dilate[:, :] == 255) & (image_color_copy_arter == 255)] = 1
                    if (len(np.unique(C_vein_dilate_judge)) == 1) & (
                            len(np.unique(C_arter_dilate_judge)) != 1) & (np.mean(VeinProb2[C_vein == 255]) < 0.6):
                        image_color[
                            (C_vein[:, :] == 255) & (image_color[:, :, 2] == 255)] = [255, 0,
                                                                                      0]

            # out artery continuity
            arter = image_color[:, :, 0]
            contours_arter, hierarchy_a = cv2.findContours(arter, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            arter_size = []
            for z in range(len(contours_arter)):
                arter_size.append(contours_arter[z].size)
            arter_size = np.sort(np.array(arter_size))
            for arter_seg in range(len(contours_arter)):
                judge_number = min(np.mean(arter_size),500)

                if contours_arter[arter_seg].size < judge_number:

                    C_arter = np.zeros(vessel.shape, np.uint8)
                    C_arter = cv2.drawContours(C_arter, contours_arter, arter_seg, (255, 255, 255), cv2.FILLED)
                    max_diameter = np.max(Skeleton(C_arter, test_use_vessel)[1])

                    image_color_copy_vein = image_color[:, :, 2].copy()
                    image_color_copy_arter = image_color[:, :, 0].copy()
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
                        4 * int(np.ceil(max_diameter)), 4 * int(np.ceil(max_diameter))))
                    image_color_copy_arter = cv2.drawContours(image_color_copy_arter, contours_arter, arter_seg,
                                                              (0, 0, 0),
                                                              cv2.FILLED)
                    C_arter_dilate = cv2.dilate(C_arter, kernel, iterations=1)
                    # image_color[(C_cross[:, :] == 255) & (image_color[:, :, 1] == 255)] = [255, 0, 0]
                    C_arter_dilate_judge = np.zeros(arter.shape, np.uint8)
                    C_arter_dilate_judge[
                        (C_arter_dilate[:, :] == 255) & (image_color_copy_arter[:, :] == 255)] = 1
                    C_vein_dilate_judge = np.zeros(arter.shape, np.uint8)
                    C_vein_dilate_judge[
                        (C_arter_dilate[:, :] == 255) & (image_color_copy_vein[:, :] == 255)] = 1

                    if (len(np.unique(C_arter_dilate_judge)) == 1) & (
                            len(np.unique(C_vein_dilate_judge)) != 1) & (np.mean(ArteryProb2[C_arter == 255]) < 0.6):
                        image_color[
                            (C_arter[:, :] == 255) & (image_color[:, :, 0] == 255)] = [0,
                                                                                       0,
                                                                                       255]
            loop=loop+1
        
        Image.fromarray(image_color).save(os.path.join(out_path, f'{image_basename[ImgNumber].split(".")[0]}.png'))

def AVclassifiation_old(out_path, PredAll1, PredAll2, VesselPredAll, DataSet=0, image_basename=''):
    """
    predAll1: predition results of artery
    predAll2: predition results of vein
    VesselPredAll: predition results of vessel
    DataSet: the length of dataset
    image_basename: the name of saved mask
    """

    ImgN = DataSet

    for ImgNumber in range(ImgN):

        height, width = PredAll1.shape[2:4]

        VesselProb = VesselPredAll[ImgNumber, 0, :, :]

        ArteryProb = PredAll1[ImgNumber, 0, :, :]
        VeinProb = PredAll2[ImgNumber, 0, :, :]

        VesselSeg = (VesselProb >= 0.1) & ((ArteryProb >= 0.2) | (VeinProb >= 0.2))
        # VesselSeg = (VesselProb >= 0.5) & ((ArteryProb >= 0.5) | (VeinProb >= 0.5))
        crossSeg = (VesselProb >= 0.1) & ((ArteryProb >= 0.6) & (VeinProb >= 0.6))
        VesselSeg = binaryPostProcessing3(VesselSeg, removeArea=100, fillArea=20)

        vesselPixels = np.where(VesselSeg > 0)

        ArteryProb2 = np.zeros((height, width))
        VeinProb2 = np.zeros((height, width))
        crossProb2 = np.zeros((height, width))
        image_color = np.zeros((3, height, width), dtype=np.uint8)
        for i in range(len(vesselPixels[0])):
            row = vesselPixels[0][i]
            col = vesselPixels[1][i]
            probA = ArteryProb[row, col]
            probV = VeinProb[row, col]
            ArteryProb2[row, col] = probA
            VeinProb2[row, col] = probV

        test_use_vessel = np.zeros((height, width), np.uint8)
        ArteryPred2 = ((ArteryProb2 >= 0.2) & (ArteryProb2 > VeinProb2))
        VeinPred2 = ((VeinProb2 >= 0.2) & (VeinProb2 > ArteryProb2))

        ArteryPred2 = binaryPostProcessing3(ArteryPred2, removeArea=100, fillArea=20)
        VeinPred2 = binaryPostProcessing3(VeinPred2, removeArea=100, fillArea=20)

        image_color[0, :, :] = ArteryPred2 * 255
        image_color[2, :, :] = VeinPred2 * 255
        image_color = image_color.transpose((1, 2, 0))

        imgBin_vessel = ArteryPred2 + VeinPred2
        imgBin_vessel[imgBin_vessel[:, :] == 2] = 1
        test_use_vessel = imgBin_vessel.copy() * 255

        vessel = cal_crosspoint(test_use_vessel)

        contours_vessel, hierarchy_c = cv2.findContours(vessel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # inter continuity
        for vessel_seg in range(len(contours_vessel)):
            C_vessel = np.zeros(vessel.shape, np.uint8)
            C_vessel = cv2.drawContours(C_vessel, contours_vessel, vessel_seg, (255, 255, 255), cv2.FILLED)
            cli = np.mean(VeinProb2[C_vessel == 255]) / np.mean(ArteryProb2[C_vessel == 255])
            if cli < 1:
                image_color[
                    (C_vessel[:, :] == 255) & (test_use_vessel[:, :] == 255)] = [255, 0, 0]
            else:
                image_color[
                    (C_vessel[:, :] == 255) & (test_use_vessel[:, :] == 255)] = [0, 0, 255]

        # out vein continuity
        vein = image_color[:, :, 2]
        contours_vein, hierarchy_b = cv2.findContours(vein, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        vein_size = []
        for z in range(len(contours_vein)):
            vein_size.append(contours_vein[z].size)
        vein_size = np.sort(np.array(vein_size))
        # image_color_copy = np.uint8(image_color).copy()
        for vein_seg in range(len(contours_vein)):
            judge_number = min(np.mean(vein_size),500)
            # cv2.putText(image_color_copy, str(vein_seg), (int(contours_vein[vein_seg][0][0][0]), int(contours_vein[vein_seg][0][0][1])), 3, 1,
            #             color=(255, 0, 0), thickness=2)
            if contours_vein[vein_seg].size < judge_number:
                C_vein = np.zeros(vessel.shape, np.uint8)
                C_vein = cv2.drawContours(C_vein, contours_vein, vein_seg, (255, 255, 255), cv2.FILLED)
                max_diameter = np.max(Skeleton(C_vein, C_vein)[1])

                image_color_copy_vein = image_color[:, :, 2].copy()
                image_color_copy_arter = image_color[:, :, 0].copy()
                # a_ori = cv2.drawContours(a_ori, contours_b, k, (0, 0, 0), cv2.FILLED)
                image_color_copy_vein = cv2.drawContours(image_color_copy_vein, contours_vein, vein_seg,
                                                         (0, 0, 0),
                                                         cv2.FILLED)
                # image_color[(C_cross[:, :] == 255) & (image_color[:, :, 1] == 255)] = [255, 0, 0]
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
                    4 * int(np.ceil(max_diameter)), 4 * int(np.ceil(max_diameter))))
                C_vein_dilate = cv2.dilate(C_vein, kernel, iterations=1)
                # cv2.imwrite(path_out_3, C_vein_dilate)
                C_vein_dilate_judge = np.zeros(vessel.shape, np.uint8)
                C_vein_dilate_judge[
                    (C_vein_dilate[:, :] == 255) & (image_color_copy_vein == 255)] = 1
                C_arter_dilate_judge = np.zeros(vessel.shape, np.uint8)
                C_arter_dilate_judge[
                    (C_vein_dilate[:, :] == 255) & (image_color_copy_arter == 255)] = 1
                if (len(np.unique(C_vein_dilate_judge)) == 1) & (
                        len(np.unique(C_arter_dilate_judge)) != 1) & (np.mean(VeinProb2[C_vein == 255]) < 0.5):
                    image_color[
                        (C_vein[:, :] == 255) & (image_color[:, :, 2] == 255)] = [255, 0,
                                                                                  0]

        # out artery continuity
        arter = image_color[:, :, 0]
        contours_arter, hierarchy_a = cv2.findContours(arter, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        arter_size = []
        for z in range(len(contours_arter)):
            arter_size.append(contours_arter[z].size)
        arter_size = np.sort(np.array(arter_size))
        for arter_seg in range(len(contours_arter)):
            judge_number = min(np.mean(arter_size),500)

            if contours_arter[arter_seg].size < judge_number:

                C_arter = np.zeros(vessel.shape, np.uint8)
                C_arter = cv2.drawContours(C_arter, contours_arter, arter_seg, (255, 255, 255), cv2.FILLED)
                max_diameter = np.max(Skeleton(C_arter, test_use_vessel)[1])

                image_color_copy_vein = image_color[:, :, 2].copy()
                image_color_copy_arter = image_color[:, :, 0].copy()
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
                    4 * int(np.ceil(max_diameter)), 4 * int(np.ceil(max_diameter))))
                image_color_copy_arter = cv2.drawContours(image_color_copy_arter, contours_arter, arter_seg,
                                                          (0, 0, 0),
                                                          cv2.FILLED)
                C_arter_dilate = cv2.dilate(C_arter, kernel, iterations=1)
                # image_color[(C_cross[:, :] == 255) & (image_color[:, :, 1] == 255)] = [255, 0, 0]
                C_arter_dilate_judge = np.zeros(arter.shape, np.uint8)
                C_arter_dilate_judge[
                    (C_arter_dilate[:, :] == 255) & (image_color_copy_arter[:, :] == 255)] = 1
                C_vein_dilate_judge = np.zeros(arter.shape, np.uint8)
                C_vein_dilate_judge[
                    (C_arter_dilate[:, :] == 255) & (image_color_copy_vein[:, :] == 255)] = 1

                if (len(np.unique(C_arter_dilate_judge)) == 1) & (
                        len(np.unique(C_vein_dilate_judge)) != 1) & (np.mean(VeinProb2[C_vein == 255]) < 0.5):
                    image_color[
                        (C_arter[:, :] == 255) & (image_color[:, :, 0] == 255)] = [0,
                                                                                   0,
                                                                                   255]

        Image.fromarray(image_color).save(os.path.join(out_path, f'{image_basename[ImgNumber].split(".")[0]}.png'))



def AVclassifiationMetrics_skeletonPixles(PredAll1,PredAll2,VesselPredAll,LabelAll1,LabelAll2,LabelVesselAll,DataSet=0, onlyMeasureSkeleton=False, strict_mode=True):
    
    """
    predAll1: predition results of artery
    predAll2: predition results of vein
    VesselPredAll: predition results of vessel
    LabelAll1: label of artery
    LabelAll2: label of vein
    LabelVesselAll: label of vessel
    DataSet: the length of dataset
    onlyMeasureSkeleton: measure skeleton
    strict_mode: strict
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
    
    for ImgNumber in range(ImgN):
        
        height, width = PredAll1.shape[2:4]
        
    
        VesselProb = VesselPredAll[ImgNumber, 0,:,:]
        VesselLabel = LabelVesselAll[ImgNumber, 0, :, :]
    
    
        ArteryLabel = LabelAll1[ImgNumber, 0, :, :]
        VeinLabel = LabelAll2[ImgNumber, 0, :, :]
    
        ArteryProb = PredAll1[ImgNumber, 0,:,:]
        VeinProb = PredAll2[ImgNumber, 0,:,:]
        
        if strict_mode:
            VesselSeg = VesselLabel
        else:
            VesselSeg = (VesselProb >= 0.1) & ((ArteryProb >= 0.2) | (VeinProb >= 0.2))
            VesselSeg= binaryPostProcessing3(VesselSeg, removeArea=100, fillArea=20)
        
        vesselPixels = np.where(VesselSeg>0)
        
        ArteryProb2 = np.zeros((height,width))
        VeinProb2 = np.zeros((height,width))

        for i in range(len(vesselPixels[0])):
            row = vesselPixels[0][i]
            col = vesselPixels[1][i]
            probA = ArteryProb[row, col]
            probV = VeinProb[row, col]
            ArteryProb2[row, col] = probA
            VeinProb2[row, col] = probV

    
        ArteryLabelImg2= ArteryLabel.copy()
        VeinLabelImg2= VeinLabel.copy()
        ArteryLabelImg2 [VesselSeg == 0] = 0
        VeinLabelImg2 [VesselSeg == 0] = 0
        ArteryVeinLabelImg = np.zeros((height, width,3), np.uint8)
        ArteryVeinLabelImg[ArteryLabelImg2>0] = (255, 0, 0)
        ArteryVeinLabelImg[VeinLabelImg2>0] = (0, 0, 255)
        ArteryVeinLabelCommon = np.bitwise_and(ArteryLabelImg2>0, VeinLabelImg2>0)
    
        if strict_mode:
            ArteryPred2 = ArteryProb2 > 0.5
            VeinPred2 = VeinProb2 >= 0.5
        else:
            ArteryPred2 = (ArteryProb2 > 0.2) & (ArteryProb2>VeinProb2)
            VeinPred2 = (VeinProb2 >= 0.2) &  (ArteryProb2<VeinProb2)

        ArteryPred2= binaryPostProcessing3(ArteryPred2, removeArea=100, fillArea=20)
        VeinPred2= binaryPostProcessing3(VeinPred2, removeArea=100, fillArea=20)

        TPimg =  np.bitwise_and(ArteryPred2>0, ArteryLabelImg2>0) # 真实为动脉，预测为动脉
        TNimg =  np.bitwise_and(VeinPred2>0, VeinLabelImg2>0) # 真实为静脉，预测为静脉
        FPimg = np.bitwise_and(ArteryPred2>0, VeinLabelImg2>0) # 真实为静脉，预测为动脉
        FPimg = np.bitwise_and(FPimg, np.bitwise_not(ArteryVeinLabelCommon))  # 真实为静脉，预测为动脉，且不属于动静脉共存区域
    
        FNimg = np.bitwise_and(VeinPred2>0, ArteryLabelImg2>0) # 真实为动脉，预测为静脉
        FNimg = np.bitwise_and(FNimg, np.bitwise_not(ArteryVeinLabelCommon)) # 真实为动脉，预测为静脉，且不属于动静脉共存区域
    
    
        if not onlyMeasureSkeleton:
            TPa = np.count_nonzero(TPimg)
            TNa = np.count_nonzero(TNimg)
            FPa = np.count_nonzero(FPimg)
            FNa = np.count_nonzero(FNimg)

            sensitivity = TPa/(TPa+FNa)
            specificity = TNa/(TNa + FPa)
            acc = (TPa + TNa) /(TPa + TNa + FPa + FNa)
            f1 = 2*TPa/(2*TPa + FPa + FNa)
            dice = 2*TPa/(2*TPa + FPa + FNa)
            iou = TPa/(TPa + FPa + FNa)
            #print('Pixel-wise Metrics', acc, sensitivity, specificity)

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
        #np.save('./tmpfile/tmp_skeleton'+str(ImgNumber)+'.npy',Skeleton)

        ArterySkeletonLabel = cv2.bitwise_and(ArteryLabelImg2, ArteryLabelImg2, mask=Skeleton)
        VeinSkeletonLabel = cv2.bitwise_and(VeinLabelImg2, VeinLabelImg2, mask=Skeleton)
    
        ArterySkeletonPred = cv2.bitwise_and(ArteryPred2, ArteryPred2, mask=Skeleton)
        VeinSkeletonPred = cv2.bitwise_and(VeinPred2, VeinPred2, mask=Skeleton)
    

        skeletonPixles = np.where(Skeleton >0)
    
        TPa_sk = 0
        TNa_sk = 0
        FPa_sk = 0
        FNa_sk = 0
        for i in range(len(skeletonPixles[0])):
            row = skeletonPixles[0][i]
            col = skeletonPixles[1][i]
            if ArterySkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1:
                TPa_sk = TPa_sk +1

            elif VeinSkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1:
                TNa_sk = TNa_sk + 1

            elif ArterySkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1\
                    and ArteryVeinLabelCommon[row, col] == 0:
                FNa_sk = FNa_sk + 1

            elif VeinSkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1\
                    and ArteryVeinLabelCommon[row, col] == 0:
                FPa_sk = FPa_sk + 1

            else:
                pass

        if  (TPa_sk+FNa_sk)==0 and (TNa_sk + FPa_sk)==0 and (TPa_sk + TNa_sk + FPa_sk + FNa_sk)==0:
            bad_case_count += 1
            bad_case_index.append(ImgNumber)
        sensitivity_sk = TPa_sk/(TPa_sk+FNa_sk)
        specificity_sk = TNa_sk/(TNa_sk + FPa_sk)
        acc_sk = (TPa_sk + TNa_sk) /(TPa_sk + TNa_sk + FPa_sk + FNa_sk)
        f1_sk = 2*TPa_sk/(2*TPa_sk + FPa_sk + FNa_sk)
        dice_sk = 2*TPa_sk/(2*TPa_sk + FPa_sk + FNa_sk)
        iou_sk = TPa_sk/(TPa_sk + FPa_sk + FNa_sk)

        senList_sk.append(sensitivity_sk)
        specList_sk.append(specificity_sk)
        accList_sk.append(acc_sk)
        f1List_sk.append(f1_sk)
        diceList_sk.append(dice_sk)
        ioulist_sk.append(iou_sk)
        #print('Skeletonal Metrics', acc_sk, sensitivity_sk, specificity_sk)
    if onlyMeasureSkeleton:
        print('Avg Skeleton Performance:', np.mean(accList_sk), np.mean(senList_sk), np.mean(specList_sk))
        return np.mean(accList_sk), np.mean(specList_sk),np.mean(senList_sk), np.mean(f1List_sk), np.mean(diceList_sk), np.mean(ioulist_sk), bad_case_index
    else:
        print('Avg Pixel-wise Performance:', np.mean(accList), np.mean(senList), np.mean(specList))
        return np.mean(accList), np.mean(specList),np.mean(senList),np.mean(f1List),np.mean(diceList),np.mean(ioulist)



if __name__ == '__main__':


    pro_path = r'F:\dw\RIP-AV\AV\log\DRIVE\running_result\ProMap_testset.npy'
    ps = np.load(pro_path)
    AVclassifiation(r'./', ps[:, 0:1, :, :], ps[:, 1:2, :, :], ps[:, 2:, :, :], DataSet=ps.shape[0], image_basename=[str(i)+'.png' for i in range(20)])
