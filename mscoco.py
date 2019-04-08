""" Routines to fetch MS COCO data.

Krzysztof Chalupka, 2017.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.transform
import skimage.color
import skimage.io as io
import numpy as np
import cv2
from pycocotools.coco import COCO

# Tell Python where the data is.
home = 'h:/COCO/'
imageDir = home + 'images/val2014'
annFile = home + 'annotations/instances_val2014.json'

# Initialize COCO api for instance annotations.
coco = COCO(annFile)
import cv2


def index_to_rgb(index):
    '''
    Find the rgb color with the class index
    :param index:
    :return: A list like [1, 2, 3]
    '''
    color_dict = {0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128], 5: [128, 0, 128],
                  6: [0, 128, 128], 7: [128, 128, 128], 8: [64, 0, 0], 9: [192, 0, 0], 10: [64, 128, 0],
                  11: [192, 128, 0], 12: [64, 0, 128], 13: [192, 0, 128], 14: [64, 128, 128], 15: [192, 128, 128],
                  16: [0, 64, 0], 17: [128, 64, 0], 18: [0, 192, 0], 19: [128, 192, 0], 20: [0, 64, 128]}
    return color_dict[index]

def gray_to_rgb(image):
    '''
    Convert the gray image(mask image) to a rgb image
    :param image: gray image, with shape [height, width]
    :return: rgb image, with shape [height, width, 3]
    '''
    height = np.shape(image)[0]
    width = np.shape(image)[1]
    result = np.zeros([height, width, 3], dtype='uint8')
    for h in range(height):
        for w in range(width):
            result[h][w] = index_to_rgb(image[h][w])
    return result

def print2d(matrix):
    for i in range(len(matrix)):
        print(matrix[i])
    print()
    print()

if __name__ == "__main__":
    # category = 'person'
    # cat_ids = coco.getCatIds()
    # cats = coco.loadCats(cat_ids)
    # for i in range(len(cats)):
    #     print(cats[i])
    voc_coco_dic = {1:5, 2:2, 3:16, 4:9, 5:44, 6:6, 7:3, 8:17, 9:62, 10:21, 11:67, 12:18, 13:19, 14:4, 15:1,
                    16:64, 17:20, 19:7, 20:72}
    coco_voc_dic = {5:1, 2:2, 16:3, 9:4, 44:5, 6:6, 3:7, 17:8, 62:9, 21:10, 67:11, 18:12, 19:13, 4:14, 1:15,
                    64:16, 20:17, 7:19, 72:20}
    img_ids = set()
    for cat_id in list(voc_coco_dic.values()):
        img_ids = img_ids | set(coco.getImgIds(catIds=cat_id))
    img_ids = list(img_ids)
    for i in range(len(img_ids)):
        image = coco.loadImgs(int(img_ids[i]))[0]
        filename = image['file_name']
        image = io.imread('{}/{}'.format(imageDir, filename))
        annIds = coco.getAnnIds(imgIds=img_ids[i])
        ann = coco.loadAnns(annIds)
        cv2.imwrite('H:/COCO/new/val2014/images/' + filename, image)
        mask = np.zeros((image.shape[0], image.shape[1]))
        for ann_single in ann:
            cat_id = ann_single['category_id']
            if cat_id not in coco_voc_dic.keys():
                continue
            single_mask = coco.annToMask(ann_single)
            single_mask = single_mask.astype(np.uint8)
            single_mask *= coco_voc_dic[cat_id]
            mask[mask == 0] += single_mask[mask == 0]
        if np.sum(mask) == 0:
            continue
        mask = mask.astype(np.uint8)
        cv2.imwrite('H:/COCO/new/val2014/annotations/' + filename[:-4] + '.png', mask)
        if i % 100 == 0:
            print(i + 1, '/', len(img_ids))
        # cv2.imshow('image', image)
        # cv2.imshow('mask', gray_to_rgb(mask))
        # cv2.waitKey(0)