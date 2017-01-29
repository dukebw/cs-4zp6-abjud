from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim



def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer):
      with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
        return arg_sc


def detection_vgg(inputs,
                          num_classes=16,
                          is_training=True,
                          dropout_keep_prob=0.5,
                          scope='detection_vgg_16'):
    """Oxford Net VGG 16-Layers Fully Convolutional with Skip-Connections as in the paper 'Fully Convolutional Networks
    for Semantic Segmentation' by Long et al.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained. This is needed to set and unset dropout
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'detection_vgg_16', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            # A1
            # Dim(inputs) = [batch_size, 224, 224, 3]
            a1 = slim.repeat(inputs, 2, slim.conv2d, num_outputs=64,kernel_size= [3, 3], stride=[1,1], scope='conv1')
            # Dim(a1) = [batch_size, 224, 224, 64]
            a1 = slim.max_pool2d(a1, [1, 1], scope='pool1')
            # Dim(a1) = [batch_size, 112, 112, 64]

            # A2
            a2 = slim.repeat(a1, 2, slim.conv2d, 128, [3, 3], [1,1], scope='conv2')
            # Dim(a2) = [batch_size, 112, 112, 128]
            a2 = slim.max_pool2d(a2, [1, 1], scope='pool2')
            # Dim(a2) = [batch_size, 56, 56, 128]

            # A3 - We send the output of this to A8 and then add it to A9
            a3 = slim.repeat(a2, 3, slim.conv2d, 256, [3, 3], [1,1], scope='conv3')
            # Dim(a3) = [batch_size, 56, 56, 256]
            a3 = slim.max_pool2d(a1, [1, 1], scope='pool3')
            # Dim(a3) = [batch_size, 28, 28, 256]

            # A4 - Also forms a skip connection - we need to send this output to A8 and then add this to A9
            a4 = slim.repeat(a3, 3, slim.conv2d, 512, [3, 3], [1,1] ,scope='conv4')
            # Dim(a4) = [batch_size, 28, 28, 512]
            a4 = slim.max_pool2d(a4, [1, 1], scope='pool4')
            # Dim(a4) = [batch_size, 14, 14, 512]

            # A5
            a5 = slim.repeat(a4, 3, slim.conv2d, 512, [1, 1], [1,1], scope='conv5')
            # Dim(a5) = [batch_size, 14, 14, 512]
            a5 = slim.max_pool2d(a5, [1, 1], scope='pool5')
            # Dim(a5) = [batch_size, 7, 7, 512]

            # Use conv2d instead of fully_connected layers.
            a6 = slim.conv2d(a5, 4096, [7, 7], [1,1], padding='SAME', scope='fc6')
            a6 = slim.dropout(a6, dropout_keep_prob, is_training=is_training,
                               scope='dropout6')
            a7 = slim.conv2d(a6, 4096, [1, 1], [1,1], scope='fc7')
            a7 = slim.dropout(a7, dropout_keep_prob, is_training=is_training,
                               scope='dropout7')
            a8 = slim.conv2d(a7, num_classes, [1, 1], [1,1],
                              activation_fn=None,
                              normalizer_fn=None,
                              scope='fc8')

            a9 = tf.image.resize_bilinear(a8, [14, 14]) # Note that this makes sense wrt a4
            a4_skip = slim.conv2d(a4, num_classes, [1,1], [1,1], activation_fn=None, normalizer_fn=None,scope='a3_skip')
            a9 = a9 + a4_skip

            a9 = tf.image.resize_bilinear(a9, [28, 28]) # Note that this makes sense wrt a5
            a3_skip = slim.conv2d(a3, num_classes, [1,1], [1,1], activation_fn=None, normalizer_fn=None,scope='a4_skip')
            a9 = a9 + a4_skip

            # This needs to be 224 x 224 so we can stack it with the input image
            a9 = tf.image.resize_bilinear(a9,[224, 224])

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return a9, end_points


def regression_net(num_classes=16,
                          is_training=True,
                          dropout_keep_prob=0.5,
                          scope='regression_vgg_16'):
    '''Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained. This is needed to set and unset dropout
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      scope: Optional scope for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    '''
    with tf.variable_scope(scope, 'regression_vgg_16', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            c1 = slim.conv2d(inputs, num_outputs=64, kernel_size=[9, 9], stride=[1,1], scope='conv1')
            c2 = slim.conv2d(c1, num_outputs=64, kernel_size=[13, 13], stride=[1,1], scope='conv2')
            c3 = slim.conv2d(c2, num_outputs=128, kernel_size=[13, 13], stride=[1,1], scope='conv3')
            c4 = slim.conv2d(c3, num_outputs=256, kernel_size=[15, 15], stride=[1,1], scope='conv4')
            c5 = slim.conv2d(c4, num_outputs=512, kernel_size=[1, 1], stride=[1,1], scope='conv5')
            c6 = slim.conv2d(c5, num_outputs=512, kernel_size=[1, 1], stride=[1,1], scope='conv6')
            c7 = slim.conv2d(c6, num_outputs=16, kernel_size=[1, 1], stride=[1,1], scope='conv7')
            c8 = slim.conv2d_transpose(c7, num_outputs=16, kernel_size=[8, 8], stride=[4, 4], scope='conv8')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return end_points
