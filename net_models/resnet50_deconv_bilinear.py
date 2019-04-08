import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
def resnet50_deconv_bilinear(input_op, num_classes, training):
    net = input_op
    net, end_points = nets.resnet_v2.resnet_v2_50(net, num_classes, training, output_stride=16, global_pool=False)
    net = slim.conv2d_transpose(net, num_outputs=64, kernel_size=3, stride=2)
    net = slim.conv2d_transpose(net, num_outputs=64, kernel_size=3, stride=2)
    net = tf.image.resize_bilinear(net, [tf.shape(input_op)[1], tf.shape(input_op)[2]])
    return net