import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
from net_models.resnet101_deconv_bilinear import resnet101_deconv_bilinear as res101db
from net_models.fcn import fcn
from net_models.pspnet import pspnet
from net_models.deeplabv3p_resnet import deeplabv3p_resnet
import os
import cv2
import time
import numpy as np
from VOC2012_no_resize import VOC2012
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
            net, var_list = deeplabv3p_resnet(net, self.num_classes, training)
        elif self.model_name == 'fcn':
            net, var_list = fcn(net, self.num_classes, training)
        elif self.model_name == 'pspnet':
            net, var_list = pspnet(net, self.num_classes, training)
        #self.logits = tf.nn.softmax(net, axis=3)
        self.var_list = var_list

        self.net = tf.argmax(net, axis=3, output_type=tf.int32)

        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
        self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net,
                                                                                   labels=self.label_op,
                                                                                   name="entropy")))
        self.loss = self.loss + reg
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.iter, decay_steps=10000,
                                                   decay_rate=0.9, staircase=False)
        # learning_rate = self.learning_rate
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

        self.train_op = slim.learning.create_train_op(self.loss, optimizer)
        #self.train_op = slim.learning.create_train_op(self.loss, optimizer,variables_to_train=decoder_vars)
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
        self.summary_op = tf.summary.merge_all()
        print('build done')
    def train(self, voc2012, START=0, MAX_ITERS=600000, restore_path='model/model.ckpt',
              base_model_path='pretrained/vgg_16/vgg_16.ckpt'):
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
        # restore or initialize

        if restore_path is None:
            saver = tf.train.Saver(var_list=self.var_list)
            #self.sess.run(tf.variables_initializer(uninit_vars))
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, base_model_path)
            print('Initialize all parameters')
            # uninit_vars_names = self.sess.run(tf.report_uninitialized_variables())
            # print(uninit_vars_names, len(uninit_vars_names))
        else:
            saver = tf.train.Saver()
            self.sess.run(tf.local_variables_initializer())
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
        saver = tf.train.Saver()
        # summary writer
        summary_writer = tf.summary.FileWriter('logs/', self.sess.graph)
        # train loop

        for iter in range(START, MAX_ITERS):
            start_time = time.time()
            # create one batch
            image_batch, label_batch = voc2012.get_batch_aug(self.batch_size)
            # train
            feed_dict = {self.input_op:image_batch, self.label_op:label_batch}
            _ = self.sess.run(self.train_op, feed_dict=feed_dict)

            loss, acc, zeros_rate = self.sess.run([self.loss, self.accuracy, self.zeros_rate], feed_dict=feed_dict)
            print('iter:', iter, 'loss:', loss, 'acc:', acc, 'zeros rate:', zeros_rate, 'time:', time.time() - start_time)

            if iter % 500 == 0 and iter > 100:
                # write summary
                image_batch, label_batch = voc2012.get_batch_val(self.batch_size)
                feed_dict = {self.input_op: image_batch, self.label_op: label_batch}
                self.sess.run(tf.local_variables_initializer())
                summary_loss, summary_acc, summary_zeros, _ = self.sess.run(
                            [self.loss_summary, self.acc_summary, self.zeros_summary, self.miou_update_op],
                                feed_dict=feed_dict)
                summary_miou = self.sess.run(self.miou_summary)
                summary_writer.add_summary(summary_loss, iter)
                summary_writer.add_summary(summary_acc, iter)
                summary_writer.add_summary(summary_zeros, iter)
                summary_writer.add_summary(summary_miou, iter)
            if iter % 5000 == 0:
                saver.save(self.sess, 'model/model.ckpt')

    def eval(self, voc2012, restore_path='model/model.ckpt'):
        '''
        Calculate mIoU score and pixel acc on voc2012 validation dataset
        :param voc2012: The VOC2012 object
        :param restore_path: The model save path
        '''
        recall, recall_update_op = tf.metrics.recall(labels=self.label_op, predictions=self.net)
        mean_acc, mean_acc_update_op = tf.metrics.mean_per_class_accuracy(labels=self.label_op,
                                            predictions=self.net, num_classes=self.num_classes)
        precision, precision_update_op = tf.metrics.precision(labels=self.label_op, predictions=self.net)
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
        for iter in range(len(voc2012.val_images)):
            image_batch, label_batch = voc2012.get_batch_val(1)
            feed_dict = {self.input_op: image_batch, self.label_op: label_batch}
            self.sess.run([self.miou_update_op, recall_update_op, mean_acc_update_op, precision_update_op],
                          feed_dict=feed_dict)
            acc = acc + self.sess.run(self.accuracy, feed_dict=feed_dict)
            print('iter:', iter + 1, '/', len(voc2012.val_images))
        acc = acc / len(voc2012.val_images)
        print('mIoU:', self.sess.run(self.miou), 'recall:', self.sess.run(recall),
              'mean acc:', self.sess.run(mean_acc), 'pixel acc:', acc, 'precision:', self.sess.run(precision))
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
            image_batch, label_batch = voc2012.get_batch_val(1)
            feed_dict = {self.input_op: image_batch, self.label_op: label_batch}
            # inference
            outputs = self.sess.run(self.net, feed_dict=feed_dict)
            cv2.imwrite(sample_path + str(i) + '.jpg', image_batch[0])
            cv2.imwrite(sample_path + str(i) + '_gt.png', label_batch[0])
            cv2.imwrite(sample_path + str(i) + '_pred.png', outputs[0])
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
