import os
from PIL import Image
import numpy as np

def mIoU(gt_path, pred_path, num_classes=21):
    filenames = os.listdir(pred_path)
    confusion_matrix = np.zeros([num_classes, num_classes])# The i class is assigned to j class
    print('Setting Confusion Matrix....')
    for i in range(len(filenames)):
        # read files
        gt = Image.open(gt_path + filenames[i])
        gt = np.array(gt)
        pred = Image.open(pred_path + filenames[i])
        pred = np.array(pred)
        # get shape
        height = np.shape(gt)[0]
        width = np.shape(pred)[1]
        # read into matrix
        for h in range(height):
            for w in range(width):
                if gt[h][w] != 255:
                    confusion_matrix[gt[h][w]][pred[h][w]] += 1

        print(i + 1, '/', len(filenames))

    # calculate miou
    print('Calculating mIoU.....')
    sum_IoU = 0
    for i in range(num_classes):# for every class
        IoU = 0
        pii = confusion_matrix[i][i]
        sum_pij = 0
        for j in range(num_classes):
            sum_pij += confusion_matrix[i][j]
        sum_pji = 0
        for j in range(num_classes):
            sum_pji += confusion_matrix[j][i]
        IoU = pii / (sum_pij + sum_pji - pii)
        sum_IoU += IoU
        print('The IoU of', i, 'th class is', IoU)
    mIoU = sum_IoU / num_classes
    print('mIoU of all classes:', mIoU)
    return mIoU




miou = mIoU(gt_path='../VOC2012/SegmentationClass/', pred_path='../val_output/')