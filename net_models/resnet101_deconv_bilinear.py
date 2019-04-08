import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
import os
def resnet101_deconv_bilinear(input_op, num_classes, training):
    net = input_op
    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
        net, end_points = nets.resnet_v2.resnet_v2_101(net, num_classes,
                            is_training=training, output_stride=16, global_pool=False)

    # restore from base model, if the path exists
    base_model = 'resnet_v2_101'
    exclude = [base_model + '/logits', 'global_step']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    #if training:
    #tf.train.init_from_checkpoint(base_model_path,
    #                                 {v.name.split(':')[0]: v for v in variables_to_restore})

    net = slim.conv2d_transpose(net, num_outputs=64, kernel_size=3, stride=2)
    net = slim.conv2d_transpose(net, num_outputs=21, kernel_size=3, stride=2)
    net = tf.image.resize_bilinear(net, [tf.shape(input_op)[1], tf.shape(input_op)[2]])
    return net, variables_to_restore