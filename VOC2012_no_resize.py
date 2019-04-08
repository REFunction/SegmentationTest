import cv2
import numpy as np
from PIL import Image
import os
import pickle
from operator import eq

def save_pickle(images, labels, path):
    print('saving', path)
    f = open(path, 'wb')
    pickle.dump([images, labels], f)
    f.close()
def load_pickle(path):
    print('loading', path)
    f = open(path, 'rb')
    images, labels = pickle.load(f)
    return images, labels
class VOC2012:
    '''
    voc2012 tool for image semantic segmentation
    Here is no resize
    All images and labels are kept at the original size
    '''
    def __init__(self, root_path='./VOC2012/', aug_path='SegmentationClassAug/',
                 checkpaths=False):
        '''
        Create a VOC2012 object
        This function will set all paths needed, do not set them mannully expect you have
        changed the dictionary structure
        Args:
            root_path:the Pascal VOC 2012 folder path
            image_size:resize images and labels into this size
        '''
        self.root_path = root_path
        if root_path[len(root_path) - 1] != '/' and root_path[len(root_path) - 1] != '\\':
            self.root_path += '/'
        self.train_list_path = self.root_path + 'ImageSets/Segmentation/train.txt'
        self.val_list_path = self.root_path + 'ImageSets/Segmentation/val.txt'
        self.image_path = self.root_path + 'JPEGImages/'
        self.label_path = self.root_path + 'SegmentationClass/'
        self.aug_path = aug_path
        if aug_path[len(aug_path) - 1] != '/' and aug_path[len(aug_path) - 1] != '\\':
            self.aug_path += '/'
        if checkpaths:
            self.check_paths()

    def check_paths(self):
        '''
        check all paths and display the status of paths
        '''
        if not (os.path.exists(self.root_path) and os.path.isdir(self.root_path)):
            print('Warning: Dictionary', self.root_path, ' does not exist')
        if not (os.path.exists(self.train_list_path) and os.path.isfile(self.train_list_path)):
            print('Warning: Training list file', self.train_list_path, 'does not exist')
        if not (os.path.exists(self.val_list_path) and os.path.isfile(self.val_list_path)):
            print('Warning: Validation list file', self.val_list_path, 'does not exist')
        if not (os.path.exists(self.image_path) and os.path.isdir(self.image_path)):
            print('Warning: Dictionary', self.image_path, 'does not exist')
        if not (os.path.exists(self.label_path) and os.path.isdir(self.label_path)):
            print('Warning: Dictionary', self.label_path, 'does not exist')
        if not (os.path.exists(self.aug_path) and os.path.isdir(self.aug_path)):
            print('Warning: Dictionary', self.aug_path, 'does not exist')

    def read_train_list(self):
        '''
        Read the filenames of training images and labels into self.train_list
        '''
        self.train_list = []
        f = open(self.train_list_path, 'r')
        line = None
        while 1:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            self.train_list.append(line)
        f.close()
    def read_val_list(self):
        '''
        Read the filenames of validation images and labels into self.val_list
        '''
        self.val_list = []
        f = open(self.val_list_path, 'r')
        line = None
        while 1:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            self.val_list.append(line)
        f.close()
        print('validation:', len(self.val_list))

    def read_train_images(self):
        '''
        Read training images into self.train_images
        If you haven't called self.read_train_list(), it will call first
        After reading images, it will resize them
        '''
        self.train_images = []
        if hasattr(self, 'train_list') == False:
            self.read_train_list()
        for filename in self.train_list:
            image = cv2.imread(self.image_path + filename + '.jpg')
            self.train_images.append(image)
            if len(self.train_images) % 100 == 0:
                print('Reading train images', len(self.train_images), '/', len(self.train_list))
    def read_train_labels(self):
        '''
        Read training labels into self.train_labels
        If you haven't called self.read_train_list(), it will call first
        After reading labels, it will resize them

        Note:image[image > 100] = 0 will remove all white borders in original labels
        '''
        self.train_labels = []
        if hasattr(self, 'train_list') == False:
            self.read_train_list()
        for filename in self.train_list:
            image = Image.open(self.label_path + filename + '.png')
            image = np.array(image)
            image[image > 20] = 0
            self.train_labels.append(image)
            if len(self.train_labels) % 100 == 0:
                print('Reading train labels', len(self.train_labels), '/', len(self.train_list))
    def read_val_images(self):
        '''
           Read validation images into self.val_images
           If you haven't called self.read_val_list(), it will call first
           After reading images, it will resize them
        '''
        self.val_images = []
        if hasattr(self, 'val_list') == False:
            self.read_val_list()
        for filename in self.val_list:
            image = cv2.imread(self.image_path + filename + '.jpg')
            self.val_images.append(image)
            if len(self.val_images) % 100 == 0:
                print('Reading val images', len(self.val_images), '/', len(self.val_list))
    def read_val_labels(self):
        '''
           Read validation labels into self.val_labels
           If you haven't called self.read_val_list(), it will call first
           After reading labels, it will resize them

           Note:image[image > 100] = 0 will remove all white borders in original labels
        '''
        self.val_labels = []
        if hasattr(self, 'val_list') == False:
            self.read_val_list()
        for filename in self.val_list:
            image = Image.open(self.label_path + filename + '.png')
            image = np.array(image)
            image[image > 20] = 0
            self.val_labels.append(image)
            if len(self.val_labels) % 100 == 0:
                print('Reading val labels', len(self.val_labels), '/', len(self.val_list))
    def read_aug_images_labels_and_save(self, save_path='./voc2012_aug.pic'):
        '''
        read augmentation images and labels, and save them in the form of '.pic'
        Note:This function will cost a great amount of memory. Leave enough to call it.
        '''
        is_continue = input('Warning:Reading augmentation files may take up a lot of memory, continue?[y/n]')

        if is_continue != 'y' and is_continue != 'Y':
            return

        if hasattr(self, 'aug_images') == False:
            self.aug_images = []
        if hasattr(self, 'aug_labels') == False:
            self.aug_labels = []
        # check
        if self.aug_path is None or os.path.exists(self.aug_path) == False:
            raise Exception('No augmentation dictionary.Set attribute \'aug_path\' first')
        if self.image_path is None or os.path.exists(self.image_path) == False:
            raise Exception('Cannot find VOC2012 images path.')

        if hasattr(self, 'val_list') == False:
            self.read_val_list()
        aug_labels_filenames = os.listdir(self.aug_path)
        print('All augmentation iamges:', len(aug_labels_filenames), len(self.val_list))
        for i in range(len(aug_labels_filenames)):
            aug_labels_filenames[i] = aug_labels_filenames[i][:-4]
        aug_labels_filenames = list(set(aug_labels_filenames) - set(self.val_list))
        for i in range(len(aug_labels_filenames)):
            aug_labels_filenames[i] = aug_labels_filenames[i] + '.png'
        print('The rest except validation:', len(aug_labels_filenames))
        for label_filename in aug_labels_filenames:
            # read label
            label = cv2.imread(self.aug_path + label_filename, cv2.IMREAD_GRAYSCALE)
            label[label > 20] = 0
            self.aug_labels.append(label)
            # read image
            image_filename = label_filename.replace('.png','.jpg')
            image = cv2.imread(self.image_path + image_filename)
            self.aug_images.append(image)
            if len(self.aug_labels) % 100 == 0:
                print('Reading augmentation image & label pairs', len(self.aug_labels), '/',
                                                                    len(aug_labels_filenames))
        self.converge_aug_data()
        save_pickle(self.aug_images, self.aug_labels, save_path)
    def calc_pixel_mean(self, dataset='voc2012_aug'):
        sum_r = 0
        sum_g = 0
        sum_b = 0
        i = 0
        for img in self.val_images:
            sum_r = sum_r + img[:, :, 0].mean()
            sum_g = sum_g + img[:, :, 1].mean()
            sum_b = sum_b + img[:, :, 2].mean()
            i += 1
            if i % 100 == 0:
                print(i)
        sum_r = sum_r / i
        sum_g = sum_g / i
        sum_b = sum_b / i
        print(sum_r, sum_g, sum_b)
    def load_aug_data(self, aug_data_path='./voc2012_aug.pic'):
        self.aug_images, self.aug_labels = load_pickle(aug_data_path)
    def save_train_data(self, path='./voc2012_train.pic'):
        '''
        save training images and labels into path in the form of .pic
        Args:
            path:The path you want to save train data into.It must be xxx.pic
        '''
        save_pickle(self.train_images, self.train_labels, path)
    def save_val_data(self, path='./voc2012_val.pic'):
        '''
        save validation images and labels into path in the form of .pic
        Args:
            path:The path you want to save train data into.It must be xxx.pic
        '''
        save_pickle(self.val_images, self.val_labels, path)
    def read_all_data_and_save(self, train_data_save_path='./voc2012_train.pic', val_data_save_path='./voc2012_val.pic'):
        '''
        Read training and validation data and save them into two .pic files.
        Args:
            train_data_save_path:The path you want to save training data into.
            val_data_save_path:The path you want to save validation data into.
        '''
        self.read_train_images()
        self.read_train_labels()
        self.read_val_images()
        self.read_val_labels()
        self.save_train_data(train_data_save_path)
        self.save_val_data(val_data_save_path)
    def converge_train_data(self):
        '''
        Converge train images and labels, the images with same size will be adjacent
        '''
        # step 1:put images of different sizes into bins
        print('Bining.....')
        bins_images = {} # every key is a shape, every value is a list, whose every element is a image
        bins_labels = {}
        for i in range(len(self.train_images)):
            train_image = self.train_images[i]
            train_label = self.train_labels[i]
            if np.shape(train_image) in bins_images.keys():
                bins_images[np.shape(train_image)].append(train_image)
                bins_labels[np.shape(train_image)].append(train_label)
            else:
                bins_images[np.shape(train_image)] = [train_image]
                bins_labels[np.shape(train_image)] = [train_label]
            if i % 100 == 0:
                print('Bining:', i, '/', len(self.train_images))
        # step 2:collect from bins
        del self.train_images # clear memory
        del self.train_labels
        self.train_images = []
        self.train_labels = []
        for size in bins_images.keys():
            self.train_images += bins_images[size]
            self.train_labels += bins_labels[size]
    def converge_val_data(self):
        '''
        Converge validation images and labels, the images with same size will be adjacent
        '''
        # step 1:put images of different sizes into bins
        print('Bining.....')
        bins_images = {} # every key is a shape, every value is a list, whose every element is a image
        bins_labels = {}
        for i in range(len(self.val_images)):
            val_image = self.val_images[i]
            val_label = self.val_labels[i]
            if np.shape(val_image) in bins_images.keys():
                bins_images[np.shape(val_image)].append(val_image)
                bins_labels[np.shape(val_image)].append(val_label)
            else:
                bins_images[np.shape(val_image)] = [val_image]
                bins_labels[np.shape(val_image)] = [val_label]
            if i % 100 == 0:
                print('Bining:', i, '/', len(self.val_images))
        # step 2:collect from bins
        del self.val_images # clear memory
        del self.val_labels
        self.val_images = []
        self.val_labels = []
        for size in bins_images.keys():
            self.val_images += bins_images[size]
            self.val_labels += bins_labels[size]
    def converge_aug_data(self):
        '''
        Converge augmentation images and labels, the images with same size will be adjacent
        '''
        # step 1:put images of different sizes into bins
        print('Bining.....')
        bins_images = {} # every key is a shape, every value is a list, whose every element is a image
        bins_labels = {}
        for i in range(len(self.aug_images)):
            aug_image = self.aug_images[i]
            aug_label = self.aug_labels[i]
            if np.shape(aug_image) in bins_images.keys():
                bins_images[np.shape(aug_image)].append(aug_image)
                bins_labels[np.shape(aug_image)].append(aug_label)
            else:
                bins_images[np.shape(aug_image)] = [aug_image]
                bins_labels[np.shape(aug_image)] = [aug_label]
            if i % 100 == 0:
                print('Bining:', i, '/', len(self.aug_images))
        # step 2:collect from bins
        del self.aug_images # clear memory
        del self.aug_labels
        self.aug_images = []
        self.aug_labels = []
        for size in bins_images.keys():
            self.aug_images += bins_images[size]
            self.aug_labels += bins_labels[size]

    def load_all_data(self, train_data_load_path='./voc2012_train.pic', val_data_load_path='./voc2012_val.pic'):
        '''
        Load training and validation data from .pic files
        Args:
            train_data_load_path:The training data .pic file path.
            val_data_load_path:The validation data .pic file path.
        '''
        self.load_train_data(train_data_load_path)
        self.load_val_data(val_data_load_path)
    def load_train_data(self, path='./voc2012_train.pic'):
        '''
        Load training data from .pic files
        Args:
            train_data_load_path:The training data .pic file path.
        '''
        self.train_images, self.train_labels = load_pickle(path)
    def load_val_data(self, path='./voc2012_val.pic'):
        '''
        Load validation data from .pic files
        Args:
            val_data_load_path:The validation data .pic file path.
        '''
        self.val_images, self.val_labels = load_pickle(path)

    def get_batch_train(self, max_batch_size):
        '''
        Get a batch data from training data.
        It maintains an internal location variable and get from start to end gradually.
        When it comes into the end, it returns to the start.
        Args:
            batch_size:The number of images or labels returns at a time.
        Return:
            batch_images:A batch of images with shape:[batch_size, image_size, image_size, 3]
            batch_labels:A batch of labels with shape:[batch_size, image_size, image_size]
        '''
        if hasattr(self, 'train_location') == False:
            self.train_location = 0
        end = self.train_location + max_batch_size
        start = self.train_location

        batch_images = self.train_images[start]
        batch_labels = self.train_labels[start]
        self.train_location = (self.train_location + 1) % len(self.train_images)

        origin_shape = np.shape(batch_images)

        batch_images = np.array(batch_images)[np.newaxis, :, :, :]
        batch_labels = np.array(batch_labels)[np.newaxis, :, :]
        for i in range(start + 1, end):
            if np.shape(self.train_images[i % len(self.train_images)]) != origin_shape: # if shape not equal
                break
            new_image = np.array(self.train_images[i % len(self.train_images)])[np.newaxis, :, :, :]
            new_label = np.array(self.train_labels[i % len(self.train_labels)])[np.newaxis, :, :]
            batch_images = np.concatenate([batch_images, new_image], axis=0)
            batch_labels = np.concatenate([batch_labels, new_label], axis=0)
            self.train_location = (self.train_location + 1) % len(self.train_images)

        return batch_images, batch_labels
    def get_batch_val(self, max_batch_size):
        '''
        Get a batch data from validation data.
        It maintains an internal location variable and get from start to end gradually.
        When it comes into the end, it returns to the start.
        Args:
            batch_size:The number of images or labels returns at a time.
        Return:
            batch_images:A batch of images with shape:[batch_size, image_size, image_size, 3]
            batch_labels:A batch of labels with shape:[batch_size, image_size, image_size]
        '''
        if hasattr(self, 'val_location') == False:
            self.val_location = 0
        end = self.val_location + max_batch_size
        start = self.val_location

        batch_images = self.val_images[start]
        batch_labels = self.val_labels[start]
        self.val_location = (self.val_location + 1) % len(self.val_images)

        origin_shape = np.shape(batch_images)

        batch_images = np.array(batch_images)[np.newaxis, :, :, :]
        batch_labels = np.array(batch_labels)[np.newaxis, :, :]
        for i in range(start + 1, end):
            if np.shape(self.val_images[i % len(self.val_images)]) != origin_shape: # if shape not equal
                break
            new_image = np.array(self.val_images[i % len(self.val_images)])[np.newaxis, :, :, :]
            new_label = np.array(self.val_labels[i % len(self.val_labels)])[np.newaxis, :, :]
            batch_images = np.concatenate([batch_images, new_image], axis=0)
            batch_labels = np.concatenate([batch_labels, new_label], axis=0)
            self.val_location = (self.val_location + 1) % len(self.val_images)

        return batch_images, batch_labels
    def get_batch_aug(self, max_batch_size):
        '''
        Get a batch data from augmentation data.
        It maintains an internal location variable and get from start to end gradually.
        When it comes into the end, it returns to the start.
        Args:
           batch_size:The number of images or labels returns at a time.
        Return:
           batch_images:A batch of images with shape:[batch_size, image_size, image_size, 3]
           batch_labels:A batch of labels with shape:[batch_size, image_size, image_size]
        '''
        if hasattr(self, 'aug_location') == False:
            self.aug_location = 0
        end = self.aug_location + max_batch_size
        start = self.aug_location

        batch_images = self.aug_images[start]
        batch_labels = self.aug_labels[start]
        self.aug_location = (self.aug_location + 1) % len(self.aug_images)

        origin_shape = np.shape(batch_images)

        batch_images = np.array(batch_images)[np.newaxis, :, :, :]
        batch_labels = np.array(batch_labels)[np.newaxis, :, :]
        for i in range(start + 1, end):
            if np.shape(self.aug_images[i % len(self.aug_images)]) != origin_shape: # if shape not equal
                break
            new_image = np.array(self.aug_images[i % len(self.aug_images)])[np.newaxis, :, :, :]
            new_label = np.array(self.aug_labels[i % len(self.aug_labels)])[np.newaxis, :, :]
            batch_images = np.concatenate([batch_images, new_image], axis=0)
            batch_labels = np.concatenate([batch_labels, new_label], axis=0)
            self.aug_location = (self.aug_location + 1) % len(self.aug_images)

        return batch_images, batch_labels


if __name__ == '__main__':
    voc2012 = VOC2012('./VOC2012', aug_path='./VOC2012/SegmentationClassAug/')
    voc2012.read_aug_images_labels_and_save()
    # voc2012.load_val_data(path='data/voc2012_val.pic')
    # voc2012.calc_pixel_mean()
    # voc2012.load_aug_data()
    # for i in range(2000):
    #     x, y = voc2012.get_batch_aug(8)
    #     print(np.shape(x), np.shape(y))
    #voc2012.read_aug_images_labels_and_save()