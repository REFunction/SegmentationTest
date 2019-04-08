import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
from math import ceil
import numpy as np

def pspnet(input_op, num_classes, training, batch_norm_decay=0.9997):
    input_shape = tf.shape(input_op)
    net = input_op
    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        net, end_points = nets.resnet_v2.resnet_v2_101(net, num_classes=None,
                                                       is_training=training, output_stride=16, global_pool=False)
        base_model = 'resnet_v2_101'
        exclude = [base_model + '/logits', 'global_step']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        net = end_points[base_model + '/block4'] # OS = 16

    net = pyramid_pooling_module(net, input_shape) # OS = 8

    #x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4", use_bias=False)(psp)
    net = slim.conv2d(net, 512, 3, stride=1, padding='SAME', biases_initializer=None, activation_fn=None)
    #x = BN(name="conv5_4_bn")(x)
    net = slim.batch_norm(net)
    #x = Activation('relu')(x)
    net = tf.nn.relu(net)
    #x = Dropout(0.1)(x)
    net = slim.dropout(net, keep_prob=0.9, is_training=training)

    #x = Conv2D(num_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    net = slim.conv2d(net, num_classes, 1, stride=1, activation_fn=None)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    # x = Interp([input_shape[0], input_shape[1]])(x)
    net = tf.image.resize_bilinear(net, [input_shape[1], input_shape[2]])
    # x = Activation('softmax')(x)

    return net, variables_to_restore
def pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = [tf.shape(res)[1], tf.shape(res)[2]]

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    # res = Concatenate()([res,
    #                      interp_block6,
    #                      interp_block3,
    #                      interp_block2,
    #                      interp_block1])
    res = tf.concat([res, interp_block6, interp_block3, interp_block2, interp_block1], axis=3)
    return res
def interp_block(prev_layer, level, feature_map_shape, input_shape):
    kernel_strides_map = {1: 16, 2: 8, 3: 4, 6: 2}

    #prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = slim.avg_pool2d(prev_layer, kernel_strides_map[level], kernel_strides_map[level])
    #prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev_layer)
    prev_layer = slim.conv2d(prev_layer, 256, 1, stride=1, biases_initializer=None, activation_fn=None)
    #prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = slim.batch_norm(prev_layer)
    #prev_layer = Activation('relu')(prev_layer)
    prev_layer = tf.nn.relu(prev_layer)
    #prev_layer = Interp(feature_map_shape)(prev_layer)
    prev_layer = tf.image.resize_bilinear(prev_layer, size=feature_map_shape)
    return prev_layer
