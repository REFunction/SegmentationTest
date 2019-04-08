import os
from PIL import Image
import numpy as np

def mAP(gt_path, pred_path, num_classes=21):# Note:This is different from VOC Evaluation Server
    filenames = os.listdir(pred_path)
    confusion_matrix = np.zeros([num_classes, num_classes])# The i class is assigned to j class
    print('Setting Confusion Matrix....')
    for i in range(len(filenames)):
        # read files
        gt = Image.open(gt_path + filenames[i])
        gt = np.array(gt)
        gt[gt > (num_classes - 1)] = 0
        pred = Image.open(pred_path + filenames[i])
        pred = np.array(pred)
        # get shape
        height = np.shape(gt)[0]
        width = np.shape(pred)[1]
        # read into matrix
        for h in range(height):
            for w in range(width):
                confusion_matrix[gt[h][w]][pred[h][w]] += 1

        print(i + 1, '/', len(filenames))

    # calculate mAP
    print('Calculating mAP.....')
    sum_AP = 0
    for i in range(num_classes):# for every class
        AP = 0
        pii = confusion_matrix[i][i]
        sum_pji = 0
        for j in range(num_classes):
            sum_pji += confusion_matrix[j][i]
        AP = pii / sum_pji
        sum_AP += AP
        print('The AP of', i, 'th class is', AP)
    mAP = sum_AP / num_classes
    print('mAP of all classes:', mAP)
    return mAP




map = mAP(gt_path='H:/VOC2012/SegmentationClass/', pred_path='../val_output/')