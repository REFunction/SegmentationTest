import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
from net_models.resnet101_deconv_bilinear import resnet101_deconv_bilinear as res101db
from net_models.deeplabv3p_resnet import deeplabv3p_resnet
from net_models.fcn import fcn
from net_models.pspnet import pspnet
from net_models.auto_deeplab import auto_deeplab
#from net_models.crf import crf
import os
import cv2
import numpy as np
import time
from VOC2012_slim import VOC2012
from VOC2012_test import VOC2012_test
from VOC2012_val import VOC2012_val
from COCO2014 import COCO2014
from tensorpack.tfutils.optimizer import *

class network:
    def __init__(self, learning_rate=3e-5, batch_size=8, num_classes=21, model_name='deeplabv3p'):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model_name = model_name
    def build_network(self, training=True):
        print('building network...')
        with tf.name_scope('input'):
            self.input_op = tf.placeholder(tf.float32, [None, None, None, 3])
            self.label_op = tf.placeholder(tf.int32, [None, None, None])
            self.iter = tf.placeholder(dtype=tf.int32)
        net = self.input_op - np.array([104.00699, 116.66877, 122.67892])

        if self.model_name == 'deeplabv3p':
            net, var_list = deeplabv3p_resnet(net, self.num_classes, training, output_stride=8)
        elif self.model_name == 'fcn':
            net, var_list = fcn(net, self.num_classes, training)
        elif self.model_name == 'pspnet':
            net, var_list = pspnet(net, self.num_classes, training)
        elif self.model_name == 'auto_deeplab':
            net = auto_deeplab(net, self.num_classes, training)
            var_list = None
        self.logits = tf.nn.softmax(net, axis=3)
        self.var_list = var_list

        self.net = tf.argmax(net, axis=3, output_type=tf.int32)

        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(2e-4), tf.trainable_variables())
        self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net,
                            labels=self.label_op,name="entropy")))
        self.loss = self.loss + reg

        #learning_rate = self.learning_rate
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = AccumGradOptimizer(optimizer, 4)
        self.train_op = slim.learning.create_train_op(self.loss, optimizer, global_step=None)
        # accuracy
        correct_prediction = tf.equal(self.net, self.label_op)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # zero rate
        self.zeros = tf.zeros_like(self.net)
        zeros_prediction = tf.equal(self.net, self.zeros)
        self.zeros_rate = tf.reduce_mean(tf.cast(zeros_prediction, tf.float32))
        #miou
        self.miou, self.miou_update_op = tf.metrics.mean_iou(tf.reshape(self.label_op, [-1]), tf.reshape(self.net, [-1]), self.num_classes)
        # summary
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.acc_summary = tf.summary.scalar("acc", self.accuracy)
        self.zeros_summary = tf.summary.scalar("zeros rate", self.zeros_rate)
        self.miou_summary = tf.summary.scalar('mIOU', self.miou)
        print('build done')

    def train(self, voc2012, START=0, MAX_ITERS=1000000, restore_path='model/model.ckpt',
              base_model_path='pretrained/resnet_v2_101/resnet_v2_101.ckpt', coco2014=None):
        '''
        Train this network
        :param voc2012: The VOC2012 object
        :param START: Iter number. If you want to restore a model and continue to train, set START to a non-zero number
        :param MAX_ITERS: The maximum of iters
        :param restore_path: The model save path. Set None if it is a new train from blank network.
        '''
        # open session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #self.sess = tf.Session()
        # restore or initialize

        if restore_path is None:
            self.sess.run(tf.global_variables_initializer())
            if self.var_list is not None:
                saver = tf.train.Saver(var_list=self.var_list)
                saver.restore(self.sess, base_model_path)
            print('Initialize all parameters')
        else:
            saver = tf.train.Saver()
            self.sess.run(tf.local_variables_initializer())
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
        saver = tf.train.Saver()
        # summary writer
        summary_writer = tf.summary.FileWriter('logs/', self.sess.graph)
        # local init
        local_init_op = tf.local_variables_initializer()
        # train loop

        for iter in range(START, MAX_ITERS):
            #self.sess.run(self.clear_grads_cache_op, feed_dict={self.iter:iter})
            start_time = time.time()
            # create one batch
            if coco2014 is not None:
                image_batch, label_batch = coco2014.get_batch_fast(self.batch_size)
            else:
                image_batch, label_batch = voc2012.get_batch_aug_fast(self.batch_size,
                                                          random_resize=False)
            # train
            feed_dict = {self.input_op: image_batch, self.label_op:label_batch, self.iter:iter}
            _, loss, acc, zeros_rate = self.sess.run([self.train_op, self.loss, self.accuracy,
                                                      self.zeros_rate], feed_dict=feed_dict)
            print('iter:', iter, 'loss:', loss, 'acc:', acc, 'zeros rate:', zeros_rate, 'time:', time.time() - start_time)

            if iter % 500 == 0 and iter > 100:
                # write summary
                image_batch, label_batch = voc2012.get_batch_val(self.batch_size)
                feed_dict = {self.input_op: image_batch, self.label_op: label_batch}
                self.sess.run(local_init_op)
                summary_loss, summary_acc, summary_zeros, _ = self.sess.run(
                            [self.loss_summary, self.acc_summary, self.zeros_summary, self.miou_update_op],
                                feed_dict=feed_dict)
                summary_miou = self.sess.run(self.miou_summary)
                summary_writer.add_summary(summary_loss, iter)
                summary_writer.add_summary(summary_acc, iter)
                summary_writer.add_summary(summary_zeros, iter)
                summary_writer.add_summary(summary_miou, iter)
            if iter % 1000 == 0:
                saver.save(self.sess, 'model/model.ckpt')

    def eval(self, voc2012, restore_path='model/model.ckpt'):
        '''
        Calculate mIoU score and pixel acc on voc2012 validation dataset
        :param voc2012: The VOC2012 object
        :param restore_path: The model save path
        '''
        # open session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # restore or initialize
        saver = tf.train.Saver()
        if restore_path is None:
            print('Error:No model restore path')
            return
        else:
            self.sess.run(tf.local_variables_initializer())
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
        print('Evaluating......')
        acc = 0
        for iter in range(len(voc2012.val_names)):
            image_batch, label_batch = voc2012.get_batch_val(1)
            feed_dict = {self.input_op: image_batch, self.label_op: label_batch}

            self.sess.run(self.miou_update_op, feed_dict=feed_dict)
            acc = acc + self.sess.run(self.accuracy, feed_dict=feed_dict)
            print('iter:', iter + 1, '/', len(voc2012.val_names))
        acc = acc / len(voc2012.val_names)
        print('mIoU:', self.sess.run(self.miou), 'pixel acc:', acc)
    def sample(self, voc2012, sample_num=8, restore_path='model/model.ckpt', sample_path='samples/'):
        '''
        Segment voc2012 validation images with models and save.
        It will inference a batch of images with network.
        :param voc2012: The VOC2012 object
        :param restore_path: The model save path
        :param sample_path: The path where you want to save images into
        '''
        # open session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # restore or initialize
        saver = tf.train.Saver()
        if restore_path is None:
            print('Error:No model restore path')
            return
        else:
            self.sess.run(tf.local_variables_initializer())
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
        print('Sampling......')
        # write into sample path
        for i in range(sample_num):
            image_batch, label_batch = voc2012.get_batch_val(8)
            feed_dict = {self.input_op: image_batch, self.label_op: label_batch}
            # inference
            outputs = self.sess.run(self.net, feed_dict=feed_dict)
            #logits = self.sess.run(self.logits, feed_dict=feed_dict)
            #outputs_crf = crf(image_batch[0], logits)
            cv2.imwrite(sample_path + str(i) + '.jpg', image_batch[0])
            cv2.imwrite(sample_path + str(i) + '_gt.png', label_batch[0])
            cv2.imwrite(sample_path + str(i) + '_pred.png', outputs[0])
            #cv2.imwrite(sample_path + str(i) + '_crf_pred.png', outputs_crf[0])
            print('iter:', i + 1, '/', sample_num)
    def segment(self, path, restore_path='model/model.ckpt'):
        '''
        Segment images from the path, the results will be saved beside the original images
        :param path: refers to an image or an image folder
        :param restore_path:The model save path
        '''
        # open session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # restore
        saver = tf.train.Saver()
        if restore_path is None:
            print('Error:No model restore path')
            return
        else:
            self.sess.run(tf.local_variables_initializer())
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
        if os.path.isfile(path):
            image = cv2.imread(path)
            image = image[np.newaxis, :, :, :] # shape:1 x size x size x 3
            feed_dict = {self.input_op:image}
            output = self.sess.run(self.net, feed_dict=feed_dict)
            cv2.imwrite(path[:-4] + '_pred.png', output[0])
            print('Save into', path[:-4] + '_pred.png')
    def test(self, voc2012_test, restore_path='model/model.ckpt', output_path='./test_output/'):
        # open session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # restore
        saver = tf.train.Saver()
        if restore_path is None:
            print('Error:No model restore path')
            return
        else:
            self.sess.run(tf.local_variables_initializer())
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
        for i in range(len(voc2012_test.images)):
            filename = voc2012_test.images[i][0]
            image = voc2012_test.images[i][1]
            image = image[np.newaxis, :, :, :]
            feed_dict = {self.input_op: image}
            output = self.sess.run(self.net, feed_dict=feed_dict)
            cv2.imwrite(output_path + filename + '.png', output[0])
            print('iter:', i + 1, '/', len(voc2012_test.images))
    def test_val(self, voc2012_val, restore_path='model/model.ckpt', output_path='./val_output/', multi_scale=False):
        # open session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # restore
        saver = tf.train.Saver()
        if restore_path is None:
            print('Error:No model restore path')
            return
        else:
            self.sess.run(tf.local_variables_initializer())
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
        for i in range(len(voc2012_val.images)):
            filename = voc2012_val.images[i][0]
            image = voc2012_val.images[i][1]
            if multi_scale:
                output = self.multi_scale_inference(image)
            else:
                image = image[np.newaxis, :, :, :]
                feed_dict = {self.input_op: image}
                output = self.sess.run(self.net, feed_dict=feed_dict)
            cv2.imwrite(output_path + filename + '.png', output[0])
            print('iter:', i + 1, '/', len(voc2012_val.images))
    def multi_scale_inference(self, image):
        height100 = np.shape(image)[0]
        width100 = np.shape(image)[1]

        height50 = int(height100 / 2)
        width50 = int(width100 / 2)

        height75 = int(height100 * 0.75)
        width75 = int(width100 * 0.75)

        image50 = cv2.resize(image, (width50, height50))[np.newaxis, :, :, :]
        image75 = cv2.resize(image, (width75, height75))[np.newaxis, :, :, :]
        image100 = image[np.newaxis, :, :, :]

        output50 = self.sess.run(self.logits, feed_dict={self.input_op:image50})
        output75 = self.sess.run(self.logits, feed_dict={self.input_op: image75})
        output100 = self.sess.run(self.logits, feed_dict={self.input_op: image100})

        output50 = cv2.resize(output50[0], dsize=(width100, height100))
        output75 = cv2.resize(output75[0], dsize=(width100, height100))
        output100 = cv2.resize(output100[0], dsize=(width100, height100))

        output = np.max(np.stack([output50, output75, output100]), axis=0)[np.newaxis, :]
        output = np.argmax(output, axis=3)
        return output