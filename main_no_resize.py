from VOC2012 import VOC2012
from network_no_resize import *
import numpy as np
import tensorflow as tf
import os

tf.flags.DEFINE_string('GPU', '0', 'Which GPU you want to use')
tf.flags.DEFINE_string('train_dataset', 'voc2012_train', 'voc2012_train/voc2012_aug')
tf.flags.DEFINE_string('val_dataset', 'voc2012_val', 'voc2012_val')
tf.flags.DEFINE_string('mode', 'train', 'train/eval/sample/segment')
tf.flags.DEFINE_string('model_path', 'model/model.ckpt', 'where the model save into')
tf.flags.DEFINE_integer('sample_num', 8, 'How many images when sampling?')
tf.flags.DEFINE_string('sample_path', 'samples/', 'Where you want to save sample images into?')
tf.flags.DEFINE_integer('batch_size', 8, 'batch size')
tf.flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate, but decays with time')
tf.flags.DEFINE_boolean('new_train', False, 'If a new train')
tf.flags.DEFINE_string('segment_path', '', 'The image or folder path you want to segment')
tf.flags.DEFINE_string('data_path', './data/', 'The location where the .h5 files saved at')
tf.flags.DEFINE_integer('start_step', 0, 'The step start from')
tf.flags.DEFINE_string('base_model_path', 'pretrained/resnet_v2_101/resnet_v2_101.ckpt',
                       'The pretrained model path')
tf.flags.DEFINE_string('model_name', 'deeplabv3p', 'deeplabv3p/fcn')



FLAGS = tf.app.flags.FLAGS

def print_config():
    print('---------------config-----------------')
    print('mode:', FLAGS.mode)
    print('GPU:', FLAGS.GPU)
    print('model name:', FLAGS.model_name)

    if FLAGS.mode == 'train':
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
    voc2012 = VOC2012()
    if FLAGS.mode == 'eval':
        net.build_network(training=False)
        voc2012.load_val_data(path=FLAGS.data_path + 'voc2012_val.pic')
        net.eval(voc2012, restore_path=FLAGS.model_path)
    elif FLAGS.mode == 'sample':
        net.build_network(training=False)
        voc2012.load_val_data(path=FLAGS.data_path + 'voc2012_val.pic')
        net.sample(voc2012, sample_num=FLAGS.sample_num, sample_path=FLAGS.sample_path,
                   restore_path=FLAGS.model_path)
    elif FLAGS.mode == 'train':
        net.build_network(training=True)
        if FLAGS.train_dataset == 'voc2012_train':
            voc2012.load_train_data(path=FLAGS.data_path + 'voc2012_train.pic')
        elif FLAGS.train_dataset == 'voc2012_aug':
            voc2012.load_aug_data(aug_data_path=FLAGS.data_path + 'voc2012_aug.pic')
        else:
            print('Unknown dataset name:', FLAGS.train_dataset)
            exit()
        voc2012.load_val_data(path=FLAGS.data_path + 'voc2012_val.pic')
        if FLAGS.new_train:
            net.train(voc2012, restore_path=None)
        else:
            net.train(voc2012, START=FLAGS.start_step, restore_path=FLAGS.model_path)
    elif FLAGS.mode == 'segment':
        net.build_network(training=False)
        net.segment(path=FLAGS.segment_path)
    else:
        print('Unknown mode:', FLAGS.mode)

if __name__ == '__main__':
    tf.app.run()