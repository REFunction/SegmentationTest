import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
from math import ceil
import numpy as np
from net_models.aspp import atrous_spatial_pyramid_pooling

def cell(pre_cell_2, pre_cell_1, training, B=5, depth=256):
    batch_norm_params = {'is_training': training}
    h_2 = pre_cell_2
    h_1 = pre_cell_1
    h = None
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        for i in range(B):
            node1_1 = slim.conv2d(h_2, depth, 5, rate=2) + slim.separable_conv2d(h_1, depth, 3, 1)
            node1_3 = slim.conv2d(h_1, depth, 3, rate=2) + slim.separable_conv2d(h_2, depth, 3, 1)
            node1_2 = slim.separable_conv2d(h_2, depth, 3, 1) + slim.separable_conv2d(node1_3, depth, 3, 1)
            node2_1 = slim.separable_conv2d(node1_1, depth, 5, 1) + slim.separable_conv2d(node1_2, depth, 5, 1)
            node2_2 = slim.conv2d(node2_1, depth, 5, rate=2) + slim.separable_conv2d(node1_3, depth, 5, 1)
            h = tf.concat([node1_1, node1_2, node1_3, node2_1, node2_2], axis=3)
            h_2 = h_1
            h_1 = h
    return h_2, h_1

def auto_deeplab(input_op, num_classes, training, depth_0=64, batch_norm_decay=0.9997):
    batch_norm_params = {'is_training': training}
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        net = slim.conv2d(input_op, depth_0, 3, stride=2) # OS = 2
        net = slim.conv2d(net, depth_0 * 2, 3, stride=2) # OS = 4
        # L1
        cell1_2, cell1_1 = cell(net, net, training, depth=depth_0 * 2) # OS = 4
        # L2
        cell2_2, cell2_1 = cell(cell1_2, cell1_1, training, depth=depth_0 * 2) # OS = 4
        # L3
        cell3_2, cell3_1 = cell(cell2_2, cell2_1, training, depth=depth_0 * 2)  # OS = 4
        # L4
        cell4_2 = slim.conv2d(cell3_2, depth_0 * 4, 3, stride=2) # OS = 8
        cell4_1 = slim.conv2d(cell3_1, depth_0 * 4, 3, stride=2) # OS = 8
        cell4_2, cell4_1 = cell(cell4_2, cell4_1, training, depth=depth_0 * 4) # OS =8
        # L5
        cell5_2 = slim.conv2d(cell4_2, depth_0 * 8, 3, stride=2) # OS = 16
        cell5_1 = slim.conv2d(cell4_1, depth_0 * 8, 3, stride=2) # OS = 16
        cell5_2, cell5_1 = cell(cell5_2, cell5_1, training, depth=depth_0 * 8) # OS = 16
        # L6
        cell6_2 = slim.conv2d(cell5_2, depth_0 * 4, 1) # OS = 16
        cell6_2 = tf.image.resize_bilinear(cell6_2, [tf.shape(cell3_2)[1], tf.shape(cell3_2)[2]]) # OS = 8
        cell6_1 = slim.conv2d(cell5_1, depth_0 * 4, 1)
        cell6_1 = tf.image.resize_bilinear(cell6_1, [tf.shape(cell3_2)[1], tf.shape(cell3_2)[2]])  # OS = 8
        cell6_2, cell6_1 = cell(cell6_2, cell6_1, training, depth=depth_0 * 4)
        # L7
        cell7_2 = slim.conv2d(cell6_2, depth_0 * 8, 3, stride=2)  # OS = 16
        cell7_1 = slim.conv2d(cell6_1, depth_0 * 8, 3, stride=2)  # OS = 16
        cell7_2, cell7_1 = cell(cell7_2, cell7_1, training, depth=depth_0 * 8)  # OS = 16
        # L8
        cell8_2, cell8_1 = cell(cell7_2, cell7_1, training, depth=depth_0 * 8) # OS = 16
        # L9
        cell9_2 = slim.conv2d(cell8_2, depth_0 * 16, 3, stride=2) # OS = 32
        cell9_1 = slim.conv2d(cell8_1, depth_0 * 16, 3, stride=2)  # OS = 32
        cell9_2, cell9_1 = cell(cell9_2, cell9_1, training, depth=depth_0 * 16) # OS = 32
        # L10
        cell10_2, cell10_1 = cell(cell9_2, cell9_1, training, depth=depth_0 * 16) # OS = 32
        # L11
        cell11_2 = slim.conv2d(cell10_2, depth_0 * 8, 1)  # OS = 32
        cell11_2 = tf.image.resize_bilinear(cell11_2, [tf.shape(cell8_2)[1], tf.shape(cell8_2)[2]])  # OS = 16
        cell11_1 = slim.conv2d(cell10_1, depth_0 * 8, 1)
        cell11_1 = tf.image.resize_bilinear(cell11_1, [tf.shape(cell8_2)[1], tf.shape(cell8_2)[2]])  # OS = 16
        cell11_2, cell11_1 = cell(cell11_2, cell11_1, training, depth=depth_0 * 8)
        # L12
        cell12_2 = slim.conv2d(cell11_2, depth_0 * 8, 1)  # OS = 32
        cell12_2 = tf.image.resize_bilinear(cell12_2, [tf.shape(cell6_2)[1], tf.shape(cell6_2)[2]])  # OS = 8
        cell12_1 = slim.conv2d(cell11_1, depth_0 * 8, 1)
        cell12_1 = tf.image.resize_bilinear(cell12_1, [tf.shape(cell6_2)[1], tf.shape(cell6_2)[2]])  # OS = 8
        cell12_2, cell12_1 = cell(cell12_2, cell12_1, training, depth=depth_0 * 8)

    net = atrous_spatial_pyramid_pooling(cell12_1, output_stride=8, batch_norm_decay=0.9997, is_training=training)
    net = tf.image.resize_bilinear(net, [tf.shape(input_op)[1], tf.shape(input_op)[2]])

    return net