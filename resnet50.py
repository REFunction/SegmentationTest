import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
from DataLoader import *
import os
import cv2
import time
from VOC2012 import VOC2012

class network:
    def __init__(self, learning_rate=1e-4, batch_size=8, image_size=224, num_classes=21, block_num=6):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.block_num = block_num
    def build_network(self, training=True):
        print('building network...')
        with tf.name_scope('input'):
            self.input_op = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])
            self.label_op = tf.placeholder(tf.int32, [None, self.image_size, self.image_size])

        with tf.name_scope('core_network'):
            net = self.input_op
            net, end_points = nets.resnet_v2.resnet_v2_50(net, self.num_classes, training, output_stride=16, global_pool=False)

            net = slim.conv2d_transpose(net, num_outputs=64, kernel_size=3, stride=2)
            net = slim.conv2d_transpose(net, num_outputs=64, kernel_size=3, stride=2)
            net = tf.image.resize_bilinear(net, [self.image_size, self.image_size])

        self.net = tf.argmax(net, axis=3, output_type=tf.int32)
        #self.net = tf.cast(self.net, tf.float32)
        # train
        #self.loss = tf.reduce_mean(-tf.reduce_sum(self.label_op_one_hot * tf.log(tf.clip_by_value(net, 1e-10, 1)), reduction_indices=1))
        self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net,
                            labels=self.label_op,name="entropy")))
        learning_rate = tf.train.exponential_decay(self.learning_rate, 100000, decay_steps=1000, decay_rate=0.98, staircase=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = slim.learning.create_train_op(self.loss, optimizer)
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
    def train(self, voc2012, START=0, MAX_ITERS=200000, restore_path='model/model.ckpt'):
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
        saver = tf.train.Saver()
        if restore_path is None:
            self.sess.run(tf.global_variables_initializer())
            print('Initialize all parameters')
        else:
            self.sess.run(tf.local_variables_initializer())
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
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

            if iter % 10 == 0:
                # write summary
                image_batch, label_batch = voc2012.get_batch_val(self.batch_size)
                feed_dict = {self.input_op: image_batch, self.label_op: label_batch}
                summary_loss, summary_acc, summary_zeros = self.sess.run(
                                [self.loss_summary, self.acc_summary, self.zeros_summary],
                                feed_dict=feed_dict)
                summary_writer.add_summary(summary_loss, iter)
                summary_writer.add_summary(summary_acc, iter)
                summary_writer.add_summary(summary_zeros, iter)
            if iter % 500 == 0:
                saver.save(self.sess, 'model/model.ckpt')
                # output = self.sess.run(self.net,
                #                     feed_dict={self.input_op: image_batch, self.label_op: label_batch})
                # cv2.imwrite('samples/' + str(iter) + '.jpg', output[0])
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
        MAX_ITER = 181
        for iter in range(MAX_ITER):
            image_batch, label_batch = voc2012.get_batch_val(self.batch_size)
            feed_dict = {self.input_op: image_batch, self.label_op: label_batch}
            self.sess.run(self.miou_update_op, feed_dict=feed_dict)
            acc = acc + self.sess.run(self.accuracy, feed_dict=feed_dict)
            print('iter:', iter + 1, '/', MAX_ITER)
        acc = acc / MAX_ITER
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
        # get batch data
        image_batch, label_batch = voc2012.get_batch_val(sample_num)
        feed_dict = {self.input_op: image_batch, self.label_op: label_batch}
        # inference
        outputs = self.sess.run(self.net, feed_dict=feed_dict)
        # write into sample path
        for i in range(sample_num):
            cv2.imwrite(sample_path + str(i) + '.jpg', image_batch[i])
            cv2.imwrite(sample_path + str(i) + '_gt.png', label_batch[i])
            cv2.imwrite(sample_path + str(i) + '_pred.png', outputs[i])
            print('iter:', i + 1, '/', sample_num)