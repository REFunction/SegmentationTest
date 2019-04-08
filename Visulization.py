import matplotlib.pyplot as plt
import os
import cv2
from VOC2012_slim import VOC2012
def visulize():
    for i in range(10000):
        # next paths
        image_path = 'samples/' + str(i) + '.jpg'
        pred_path = 'samples/' + str(i) + '_pred.png'
        gt_path = 'samples/' + str(i) + '_gt.png'
        if not os.path.exists(image_path):
            break
        # open
        image = plt.imread(image_path)
        pred = plt.imread(pred_path)
        gt = plt.imread(gt_path)
        # show
        sub = plt.subplot(1, 3 ,1)
        sub.set_title('Image')
        plt.imshow(image)

        sub = plt.subplot(1, 3 ,2)
        sub.set_title('Prediction')
        plt.imshow(pred)

        sub = plt.subplot(1, 3, 3)
        sub.set_title('Ground Truth')
        plt.imshow(gt)

        plt.show()
def visulize_with_crf():
    for i in range(10000):
        # next paths
        image_path = 'samples/' + str(i) + '.jpg'
        pred_path = 'samples/' + str(i) + '_pred.png'
        gt_path = 'samples/' + str(i) + '_gt.png'
        crf_path = 'samples/' + str(i) + '_crf_pred.png'
        if not os.path.exists(image_path):
            break
        # open
        image = plt.imread(image_path)
        pred = plt.imread(pred_path)
        gt = plt.imread(gt_path)
        crf = plt.imread(crf_path)
        # show
        sub = plt.subplot(2, 2 ,1)
        sub.set_title('Image')
        plt.imshow(image)

        sub = plt.subplot(2, 2 ,2)
        sub.set_title('Prediction')
        plt.imshow(pred)

        sub = plt.subplot(2, 2, 3)
        sub.set_title('Ground Truth')
        plt.imshow(gt)

        sub = plt.subplot(2, 2, 4)
        sub.set_title('crf')
        plt.imshow(crf)

        plt.show()
def visulize_one(path):
    pred = plt.imread(path)
    plt.imshow(pred)
    plt.show()
def visulize_val(path):
    filenames = os.listdir(path)
    for i in range(len(filenames)):
        label = plt.imread(path + filenames[i], cv2.IMREAD_GRAYSCALE)
        plt.imshow(label)
        plt.show()

if __name__ == '__main__':
    #visulize_val('val_output/')
    visulize()
    #visulize_with_crf()
    #visulize_one('examples/example_pred.png')

    # img = cv2.imread('examples/example_pred.png', cv2.IMREAD_GRAYSCALE)
    # result = VOC2012.gray_to_rgb(img)
    # cv2.imwrite('examples/example_pred.jpg', result)