import cv2
import numpy as np
from PIL import Image
import h5py
import pickle
import os
import shutil

def save_pickle(images, path):
    print('saving', path)
    f = open(path, 'wb')
    pickle.dump(images, f)
    f.close()
def load_pickle(path):
    print('loading', path)
    f = open(path, 'rb')
    images = pickle.load(f)
    return images
class VOC2012_val:
    def __init__(self, root_path='./VOC2012/'):
        self.images_list_path = root_path + 'ImageSets/Segmentation/val.txt'
        self.image_folder_path = root_path + 'JPEGImages/'
    def read_images_list(self):
        '''
        Read the filenames of test images into self.images_list
        '''
        self.images_list = []
        f = open(self.images_list_path, 'r')
        line = None
        while 1:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            self.images_list.append(line)
        f.close()
    def read_images(self):
        self.images = []
        if hasattr(self, 'images_list') == False:
            self.read_images_list()
        for filename in self.images_list:
            image = cv2.imread(self.image_folder_path + filename + '.jpg')
            self.images.append([filename, image])
            if len(self.images) % 100 == 0:
                print('Reading images', len(self.images), '/', len(self.images_list))
    def save_images(self, save_path='./voc2012_val.pic'):
        save_pickle(self.images, save_path)
    def load_images(self, load_path='./voc2012_val.pic'):
        self.images = load_pickle(load_path)

if __name__ == '__main__':
    voc2012_val = VOC2012_val('H:/VOC2012/')
    voc2012_val.read_images_list()
    for i in range(len(voc2012_val.images_list)):
        shutil.copy(voc2012_val.image_folder_path +
                    voc2012_val.images_list[i] + '.jpg',
                    'H:/voc2012_val_images/')
    #voc2012_test.load_images()