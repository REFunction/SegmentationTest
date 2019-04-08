from VOC2012_slim import VOC2012
from network import *
import numpy as np
import tensorflow as tf
import os
from VOC2012_test import VOC2012_test
from VOC2012_val import VOC2012_val

tf.flags.DEFINE_string('GPU', '0', 'Which GPU you want to use')
tf.flags.DEFINE_string('train_dataset', 'voc2012_train', 'voc2012_train/voc2012_aug/coco2014')
tf.flags.DEFINE_string('val_dataset', 'voc2012_val', 'voc2012_val')
tf.flags.DEFINE_string('coco_image_path', 'COCO/trainval2014/images/', '')
tf.flags.DEFINE_string('coco_label_path', 'COCO/trainval2014/annotations/', '')
tf.flags.DEFINE_string('mode', 'train', 'train/eval/sample/segment/test/test_val')
tf.flags.DEFINE_string('model_path', 'model/model.ckpt', 'where the model save into')
tf.flags.DEFINE_string('model_name', 'deeplabv3p', 'deeplabv3p/fcn/pspnet')
tf.flags.DEFINE_integer('sample_num', 8, 'How many images when sampling?')
tf.flags.DEFINE_string('sample_path', 'samples/', 'Where you want to save sample images into?')
tf.flags.DEFINE_integer('batch_size', 8, 'batch size')
tf.flags.DEFINE_float('learning_rate', 7e-3, 'Initial learning rate, but decays with time')
tf.flags.DEFINE_boolean('new_train', False, 'If a new train')
tf.flags.DEFINE_boolean('ms', False, 'If use multi scale')
tf.flags.DEFINE_string('segment_path', '', 'The image or folder path you want to segment')
tf.flags.DEFINE_string('data_path', './data/', 'The location where the .h5 files saved at')
tf.flags.DEFINE_integer('start_step', 0, 'The step start from')
tf.flags.DEFINE_string('base_model_path', 'pretrained/resnet_v2_101/resnet_v2_101.ckpt',
                       'The pretrained model path')
tf.flags.DEFINE_string('optimizer', 'adam', 'adam/momentum')



FLAGS = tf.app.flags.FLAGS


def print_config():
    print('---------------config-----------------')
    print('mode:', FLAGS.mode)
    print('GPU:', FLAGS.GPU)
    print('model name:', FLAGS.model_name)
    if FLAGS.mode == 'train':
        print('optimizer:', FLAGS.optimizer)
        print('new train:', FLAGS.new_train)
        print('train dataset:', FLAGS.train_dataset)
        print('learning rate:', FLAGS.learning_rate)
        print('batch size:', FLAGS.batch_size)
        print('data path:', FLAGS.data_path)
        if FLAGS.new_train:
            print('base model path:', FLAGS.base_model_path)
        else:
            print('model path:', FLAGS.model_path)
            print('start step:', FLAGS.start_step)
    elif FLAGS.mode == 'eval':
        print('validation dataset:', FLAGS.val_dataset)
        print('data path:', FLAGS.data_path)
    elif FLAGS.mode == 'sample':
        print('sample number:', FLAGS.sample_num)
        print('sample path:', FLAGS.sample_path)
    elif FLAGS.mode == 'segment':
        print('segment path:', FLAGS.segment_path)
    print('--------------------------------------')


def main(_):
    print_config()
    # set GPU id

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU
    # build network
    net = network(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size,
                  model_name=FLAGS.model_name)
    # create dataset
    voc2012 = VOC2012(image_size=(513, 513))
    coco2014 = None
    if FLAGS.mode == 'eval':
        net.build_network(training=False)
        voc2012.image_size = None
        #voc2012.load_val_data(path=FLAGS.data_path + 'voc2012_val.h5')
        net.eval(voc2012, restore_path=FLAGS.model_path)
    elif FLAGS.mode == 'sample':
        net.build_network(training=False)
        net.sample(voc2012, sample_num=FLAGS.sample_num, sample_path=FLAGS.sample_path,
                   restore_path=FLAGS.model_path)
    elif FLAGS.mode == 'train':
        net.build_network(training=True)
        # if FLAGS.train_dataset == 'voc2012_train':
        #     voc2012.load_train_data(path=FLAGS.data_path + 'voc2012_train.h5')
        # elif FLAGS.train_dataset == 'voc2012_aug':
        #     voc2012.load_aug_data(aug_data_path=FLAGS.data_path + 'voc2012_aug.h5')
        if FLAGS.train_dataset == 'coco2014':
            coco2014 = COCO2014(FLAGS.data_path + FLAGS.coco_image_path
                                , FLAGS.data_path + FLAGS.coco_label_path)
        if FLAGS.new_train:
            net.train(voc2012, restore_path=None, base_model_path=FLAGS.base_model_path, coco2014=coco2014)
        else:
            net.train(voc2012, START=FLAGS.start_step, restore_path=FLAGS.model_path,
                                            base_model_path = FLAGS.base_model_path, coco2014=coco2014)
    elif FLAGS.mode == 'segment':
        net.build_network(training=False)
        net.segment(path=FLAGS.segment_path, restore_path=FLAGS.model_path)
    elif FLAGS.mode == 'test':
        net.build_network(training=False)
        voc2012_test = VOC2012_test()
        voc2012_test.load_images(FLAGS.data_path + 'voc2012_test.pic')
        net.test(voc2012_test, restore_path=FLAGS.model_path)
    elif FLAGS.mode == 'test_val':
        net.build_network(training=False)
        voc2012_val = VOC2012_val()
        voc2012_val.load_images(FLAGS.data_path + 'voc2012_val.pic')
        net.test_val(voc2012_val, restore_path=FLAGS.model_path, multi_scale=FLAGS.ms)
    else:
        print('Unknown mode:', FLAGS.mode)

if __name__ == '__main__':
    tf.app.run()