# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_16
"""
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


def vgg_16_base(inputs, num_classes, dropout_keep_prob, is_training):
    """Base of the VGG-16 network (fully convolutional).

    Note that this function only constructs the network, and needs to be called
    with the correct variable scopes and tf.slim arg scopes.
    """
    a1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    a1 = slim.max_pool2d(a1, [2, 2], scope='pool1')
    a2 = slim.repeat(a1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    a2 = slim.max_pool2d(a2, [2, 2], scope='pool2')
    a3 = slim.repeat(a2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    a3 = slim.max_pool2d(a3, [2, 2], scope='pool3')
    a4 = slim.repeat(a3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    a4 = slim.max_pool2d(a4, [2, 2], scope='pool4')
    a5 = slim.repeat(a4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    a5 = slim.max_pool2d(a5, [2, 2], scope='pool5')

    # Use conv2d instead of fully_connected layers.
    a6 = slim.conv2d(a5, 4096, [7, 7], scope='fc6')
    a6 = slim.dropout(a6, dropout_keep_prob, is_training=is_training,
                      scope='dropout6')
    a7 = slim.conv2d(a6, 4096, [1, 1], scope='fc7')
    a7 = slim.dropout(a7, dropout_keep_prob, is_training=is_training,
                      scope='dropout7')

    return slim.conv2d(a7,
                       num_classes,
                       [1, 1],
                       activation_fn=None,
                       normalizer_fn=None,
                       scope='fc8')


def vgg_16_fcn32(inputs,
                 num_classes=16,
                 is_training=True,
                 dropout_keep_prob=0.5,
                 scope='vgg_16'):
    """This network doesn't use any skip layers, and directly does a bilinear
    upsample from the vgg_16/fc8 activations.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            a8 = vgg_16_base(inputs,
                             num_classes,
                             dropout_keep_prob,
                             is_training)
            a9 = tf.image.resize_bilinear(images=a8,
                                          size=inputs.get_shape()[1:3])

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return a9, end_points


def vgg_16(inputs,
           num_classes=16,
           is_training=True,
           dropout_keep_prob=0.5,
           scope='vgg_16'):
    """Oxford Net VGG 16-Layers version D Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.

    The network has been transformed to match the VGG FCN-8 from Fully
    Convolutional Networks for Semantic Segmentation, Long and Shelhamer.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes (joints).
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      scope: Optional scope for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            a8 = vgg_16_base(inputs,
                             num_classes,
                             dropout_keep_prob,
                             is_training)

            a9 = tf.image.resize_bilinear(images=a8, size=a4.get_shape()[1:3])
            # TODO(brendan): Try zero-initializing the skip layers, and compare
            # with default Xavier initializer.
            skip_a4 = slim.conv2d(a4, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='skip_a4')
            a9 = a9 + skip_a4

            a9 = tf.image.resize_bilinear(images=a9, size=a3.get_shape()[1:3])
            skip_a3 = slim.conv2d(a3, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='skip_a3')
            a9 = a9 + skip_a3

            a9 = tf.image.resize_bilinear(images=a9, size=inputs.get_shape()[1:3])

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return a9, end_points
vgg_16.default_image_size = 380
