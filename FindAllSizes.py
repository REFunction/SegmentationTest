import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


root_path = 'H:/VOC2012/SegmentationClass/'
sizes = {}
filenames = os.listdir(root_path)
all_points_num = 0
bkg_points_num = 0
for filename in filenames:
    label = cv2.imread(root_path + filename, cv2.IMREAD_GRAYSCALE)
    height = np.shape(label)[0]
    width = np.shape(label)[1]
    all_points_num += 500 * 500
    for i in range(height):
        for j in range(width):
            if label[i][j] == 0 or label[i][j] > 100:
                bkg_points_num += 1
    bkg_points_num += (500 * 500 - height * width)
    print(100 * bkg_points_num / all_points_num)