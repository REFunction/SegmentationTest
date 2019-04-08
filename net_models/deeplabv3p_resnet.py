import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
from math import ceil
import numpy as np
from net_models.upsample import upscore_layer
from net_models.aspp import atrous_spatial_pyramid_pooling


def deeplabv3p_resnet(input_op, num_classes, training, output_stride=16, batch_norm_decay=0.9997):
    net = input_op
    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        net, end_points = nets.resnet_v2.resnet_v2_101(net, num_classes=None,
                            is_training=training, output_stride=16, global_pool=False)
        base_model = 'resnet_v2_101'
        exclude = [base_model + '/logits', 'global_step']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        inputs_size = tf.shape(input_op)[1:3]
        net = end_points[base_model + '/block4']
        encoder_output = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, training)

        with tf.variable_scope("decoder"):
            with tf.contrib.slim.arg_scope(nets.resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                with slim.arg_scope([slim.batch_norm], is_training=training):
                    with tf.variable_scope("low_level_features"):
                        low_level_features = end_points[base_model + '/block1/unit_3/bottleneck_v2/conv1']
                        low_level_features = slim.conv2d(low_level_features, 48,
                                                               1, stride=1, scope='conv_1x1')
                        low_level_features_size = tf.shape(low_level_features)[1:3]

                    with tf.variable_scope("upsampling_logits"):
                        net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
                        net = tf.concat([net, low_level_features], axis=3, name='concat')
                        net = slim.conv2d(net, 256, 3, stride=1, scope='conv_3x3_1')
                        net = slim.conv2d(net, 256, 3, stride=1, scope='conv_3x3_2')
                        net = slim.conv2d(net, num_classes, 1, activation_fn=None, normalizer_fn=None,
                                                scope='conv_1x1')
                        logits = tf.image.resize_bilinear(net, inputs_size, name='upsample_2')

        return logits, variables_to_restore
