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
from tensorflow.python.framework import ops

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
                      biases_initializer=tf.zeros_initializer()):
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

    a8  = slim.conv2d(a7, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

    return a3, a4, a8


def _vgg_16(inputs,
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
    with tf.variable_scope(None, 'vgg_16', [inputs]) as sc:
        with tf.name_scope(scope):
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                a3, a4, a8 = vgg_16_base(inputs,
                                         num_classes,
                                         dropout_keep_prob,
                                         is_training)

                a9 = tf.image.resize_bilinear(images=a8, size=a4.get_shape()[1:3])
                skip_a4 = slim.conv2d(a4, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      weights_initializer=tf.zeros_initializer(),
                                      scope='skip_a4')
                a9 = a9 + skip_a4

                a9 = tf.image.resize_bilinear(images=a9, size=a3.get_shape()[1:3])
                skip_a3 = slim.conv2d(a3, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      weights_initializer=tf.zeros_initializer(),
                                      scope='skip_a3')
                a9 = a9 + skip_a3

                a9 = tf.image.resize_bilinear(images=a9, size=inputs.get_shape()[1:3])

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                return a9, end_points


def vgg_16(inputs,
           num_classes=16,
           is_detector_training=True,
           is_regressor_training=True,
           dropout_keep_prob=0.5,
           batch_norm_var_collection='moving_vars',
           scope='vgg_16'):
    """This is a wrapper around the `_vgg_16`, which assumes that we want to
    run VGG-16 as a single network pose estimator (i.e. just the detector).
    """
    return _vgg_16(inputs=inputs,
                   num_classes=num_classes,
                   is_training=is_detector_training,
                   scope=scope)


def _vgg_16_bn_relu(inputs,
                    num_classes=16,
                    is_training=True,
                    dropout_keep_prob=0.5,
                    batch_norm_var_collection='moving_vars',
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
    # We need to write notes about this.
    batch_norm_params = {
      # Decay for moving averages
      'decay': 0.9997,
      # epsilon to prevent 0s in variance
      'epsilon': 0.001,
      # Collection containing update_ops
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
      # Collection containing the moving mean and moving variance.
      'variables_collections':{
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      },
      'is_training': is_training
    }

    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        with tf.name_scope(scope):
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                # nested argscope because we don't apply activations and normalization to maxpool
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params=batch_norm_params):

                    a3, a4, a8 = vgg_16_base(inputs,
                                             num_classes,
                                             dropout_keep_prob,
                                             is_training)

                    a9 = tf.image.resize_bilinear(images=a8, size=a4.get_shape()[1:3])
                    skip_a4 = slim.conv2d(a4, num_classes, [1, 1],
                                          scope='skip_a4')
                    a9 = a9 + skip_a4

                    a9 = tf.image.resize_bilinear(images=a9, size=a3.get_shape()[1:3])
                    skip_a3 = slim.conv2d(a3, num_classes, [1, 1],
                                          scope='skip_a3')
                    a9 = a9 + skip_a3

                    a9 = tf.image.resize_bilinear(images=a9, size=inputs.get_shape()[1:3])

                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                return a9, end_points


def _vgg_bulat_regression(inputs,
                          num_classes=16,
                          is_training=True,
                          dropout_keep_prob=0.5,
                          batch_norm_var_collection='moving_vars_vgg_bulat',
                          scope='vgg_bulat_regression'):
    """An interpretation of the regression subnetwork from the Bulat paper.

    Note that dropout has been added after the two layers of size 1x1, and
    batch normalization has been added.
    """
    batch_norm_params = {
      'decay': 0.9997,
      'epsilon': 0.001,
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
      'variables_collections':{
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      },
      'is_training': is_training
    }

    with tf.variable_scope(None, 'vgg_bulat_regression', [inputs]) as sc:
        with tf.name_scope(scope):
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                outputs_collections=end_points_collection):
                net = slim.conv2d(inputs=inputs,
                                  num_outputs=64,
                                  kernel_size=[9, 9],
                                  stride=1,
                                  scope='c1')

                net = slim.conv2d(inputs=net,
                                  num_outputs=64,
                                  kernel_size=[13, 13],
                                  stride=2,
                                  scope='c2')

                net = slim.conv2d(inputs=net,
                                  num_outputs=128,
                                  kernel_size=[13, 13],
                                  stride=2,
                                  scope='c3')

                net = slim.conv2d(inputs=net,
                                  num_outputs=256,
                                  kernel_size=[15, 15],
                                  stride=1,
                                  scope='c4')

                net = slim.conv2d(inputs=net,
                                  num_outputs=512,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  scope='c5')

                net = slim.dropout(inputs=net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_training,
                                   scope='c5_dropout')

                net = slim.conv2d(inputs=net,
                                  num_outputs=512,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  scope='c6')

                net = slim.dropout(inputs=net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_training,
                                   scope='c6_dropout')

                net = slim.conv2d(inputs=net,
                                  num_outputs=16,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='c7')

                net = tf.image.resize_bilinear(images=net,
                                               size=inputs.get_shape()[1:3],
                                               name='c8')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


def _vgg_bulat_regression_maxpool_c2c3(inputs,
                                       num_classes=16,
                                       is_training=True,
                                       dropout_keep_prob=0.5,
                                       batch_norm_var_collection='moving_vars_vgg_bulat',
                                       scope='vgg_bulat_regression'):
    """An interpretation of the regression subnetwork from the Bulat paper.

    Note that dropout has been added after the two layers of size 1x1, and
    batch normalization has been added.
    """
    batch_norm_params = {
      'decay': 0.9997,
      'epsilon': 0.001,
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
      'variables_collections':{
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      },
      'is_training': is_training
    }

    with tf.variable_scope(None, 'vgg_bulat_regression', [inputs]) as sc:
        with tf.name_scope(scope):
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                outputs_collections=end_points_collection):
                net = slim.conv2d(inputs=inputs,
                                  num_outputs=64,
                                  kernel_size=[9, 9],
                                  stride=1,
                                  scope='c1')

                net = slim.conv2d(inputs=net,
                                  num_outputs=64,
                                  kernel_size=[13, 13],
                                  stride=1,
                                  scope='c2')

                net = slim.max_pool2d(inputs=net,
                                      kernel_size=[2, 2],
                                      scope='c2_pool')

                net = slim.conv2d(inputs=net,
                                  num_outputs=128,
                                  kernel_size=[13, 13],
                                  stride=1,
                                  scope='c3')

                net = slim.max_pool2d(inputs=net,
                                      kernel_size=[2, 2],
                                      scope='c3_pool')

                net = slim.conv2d(inputs=net,
                                  num_outputs=256,
                                  kernel_size=[15, 15],
                                  stride=1,
                                  scope='c4')

                net = slim.conv2d(inputs=net,
                                  num_outputs=512,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  scope='c5')

                net = slim.dropout(inputs=net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_training,
                                   scope='c5_dropout')

                net = slim.conv2d(inputs=net,
                                  num_outputs=512,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  scope='c6')

                net = slim.dropout(inputs=net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_training,
                                   scope='c6_dropout')

                net = slim.conv2d(inputs=net,
                                  num_outputs=16,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='c7')

                net = tf.image.resize_bilinear(images=net,
                                               size=inputs.get_shape()[1:3],
                                               name='c8')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


def _vgg_bulat_regression_maxpool_c3c4(inputs,
                                       num_classes=16,
                                       is_training=True,
                                       dropout_keep_prob=0.5,
                                       batch_norm_var_collection='moving_vars_vgg_bulat',
                                       scope='vgg_bulat_regression'):
    """An interpretation of the regression subnetwork from the Bulat paper.

    Note that dropout has been added after the two layers of size 1x1, and
    batch normalization has been added.
    """
    batch_norm_params = {
      'decay': 0.9997,
      'epsilon': 0.001,
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
      'variables_collections':{
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      },
      'is_training': is_training
    }

    with tf.variable_scope(None, 'vgg_bulat_regression', [inputs]) as sc:
        with tf.name_scope(scope):
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                outputs_collections=end_points_collection):
                net = slim.conv2d(inputs=inputs,
                                  num_outputs=64,
                                  kernel_size=[9, 9],
                                  stride=1,
                                  scope='c1')

                net = slim.conv2d(inputs=net,
                                  num_outputs=64,
                                  kernel_size=[13, 13],
                                  stride=1,
                                  scope='c2')

                net = slim.conv2d(inputs=net,
                                  num_outputs=128,
                                  kernel_size=[13, 13],
                                  stride=1,
                                  scope='c3')

                net = slim.max_pool2d(inputs=net,
                                      kernel_size=[2, 2],
                                      scope='c3_pool')

                net = slim.conv2d(inputs=net,
                                  num_outputs=256,
                                  kernel_size=[15, 15],
                                  stride=1,
                                  scope='c4')

                net = slim.max_pool2d(inputs=net,
                                      kernel_size=[2, 2],
                                      scope='c4_pool')

                net = slim.conv2d(inputs=net,
                                  num_outputs=512,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  scope='c5')

                net = slim.dropout(inputs=net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_training,
                                   scope='c5_dropout')

                net = slim.conv2d(inputs=net,
                                  num_outputs=512,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  scope='c6')

                net = slim.dropout(inputs=net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_training,
                                   scope='c6_dropout')

                net = slim.conv2d(inputs=net,
                                  num_outputs=16,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='c7')

                net = tf.image.resize_bilinear(images=net,
                                               size=inputs.get_shape()[1:3],
                                               name='c8')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


def _vgg_bulat_regression_maxpool_c2c3c4(inputs,
                                         num_classes=16,
                                         is_training=True,
                                         dropout_keep_prob=0.5,
                                         batch_norm_var_collection='moving_vars_vgg_bulat',
                                         scope='vgg_bulat_regression'):
    """An interpretation of the regression subnetwork from the Bulat paper.

    Note that dropout has been added after the two layers of size 1x1, and
    batch normalization has been added.
    """
    batch_norm_params = {
      'decay': 0.9997,
      'epsilon': 0.001,
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
      'variables_collections':{
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      },
      'is_training': is_training
    }

    with tf.variable_scope(None, 'vgg_bulat_regression', [inputs]) as sc:
        with tf.name_scope(scope):
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                outputs_collections=end_points_collection):
                net = slim.conv2d(inputs=inputs,
                                  num_outputs=64,
                                  kernel_size=[9, 9],
                                  stride=1,
                                  scope='c1')

                net = slim.conv2d(inputs=net,
                                  num_outputs=64,
                                  kernel_size=[13, 13],
                                  stride=1,
                                  scope='c2')

                net = slim.max_pool2d(inputs=net,
                                      kernel_size=[2, 2],
                                      scope='c2_pool')

                net = slim.conv2d(inputs=net,
                                  num_outputs=128,
                                  kernel_size=[13, 13],
                                  stride=1,
                                  scope='c3')

                net = slim.max_pool2d(inputs=net,
                                      kernel_size=[2, 2],
                                      scope='c3_pool')

                net = slim.conv2d(inputs=net,
                                  num_outputs=256,
                                  kernel_size=[15, 15],
                                  stride=1,
                                  scope='c4')

                net = slim.max_pool2d(inputs=net,
                                      kernel_size=[2, 2],
                                      scope='c4_pool')

                net = slim.conv2d(inputs=net,
                                  num_outputs=512,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  scope='c5')

                net = slim.dropout(inputs=net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_training,
                                   scope='c5_dropout')

                net = slim.conv2d(inputs=net,
                                  num_outputs=512,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  scope='c6')

                net = slim.dropout(inputs=net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_training,
                                   scope='c6_dropout')

                net = slim.conv2d(inputs=net,
                                  num_outputs=16,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='c7')

                net = tf.image.resize_bilinear(images=net,
                                               size=inputs.get_shape()[1:3],
                                               name='c8')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


def _vgg_bulat_regression_conv3x3_c2c3c4(inputs,
                                         num_classes=16,
                                         is_training=True,
                                         dropout_keep_prob=0.5,
                                         batch_norm_var_collection='moving_vars_vgg_bulat',
                                         scope='vgg_bulat_regression'):
    """An interpretation of the regression subnetwork from the Bulat paper.

    Note that dropout has been added after the two layers of size 1x1, and
    batch normalization has been added.
    """
    batch_norm_params = {
      'decay': 0.9997,
      'epsilon': 0.001,
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
      'variables_collections':{
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      },
      'is_training': is_training
    }

    with tf.variable_scope(None, 'vgg_bulat_regression', [inputs]) as sc:
        with tf.name_scope(scope):
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                outputs_collections=end_points_collection):
                net = slim.conv2d(inputs=inputs,
                                  num_outputs=64,
                                  kernel_size=[9, 9],
                                  stride=1,
                                  scope='c1')

                net = slim.conv2d(inputs=net,
                                  num_outputs=64,
                                  kernel_size=[13, 13],
                                  stride=1,
                                  scope='c2')

                net = slim.conv2d(inputs=net,
                                  num_outputs=64,
                                  kernel_size=[3, 3],
                                  stride=2,
                                  scope='c2_pool')

                net = slim.conv2d(inputs=net,
                                  num_outputs=128,
                                  kernel_size=[13, 13],
                                  stride=1,
                                  scope='c3')

                net = slim.conv2d(inputs=net,
                                  num_outputs=128,
                                  kernel_size=[3, 3],
                                  stride=2,
                                  scope='c3_pool')

                net = slim.conv2d(inputs=net,
                                  num_outputs=256,
                                  kernel_size=[15, 15],
                                  stride=1,
                                  scope='c4')

                net = slim.conv2d(inputs=net,
                                  num_outputs=256,
                                  kernel_size=[3, 3],
                                  stride=2,
                                  scope='c4_pool')

                net = slim.conv2d(inputs=net,
                                  num_outputs=512,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  scope='c5')

                net = slim.dropout(inputs=net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_training,
                                   scope='c5_dropout')

                net = slim.conv2d(inputs=net,
                                  num_outputs=512,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  scope='c6')

                net = slim.dropout(inputs=net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_training,
                                   scope='c6_dropout')

                net = slim.conv2d(inputs=net,
                                  num_outputs=16,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='c7')

                net = tf.image.resize_bilinear(images=net,
                                               size=inputs.get_shape()[1:3],
                                               name='c8')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


def vgg_16_bn_relu(inputs,
                   num_classes=16,
                   is_detector_training=True,
                   is_regressor_training=True,
                   dropout_keep_prob=0.5,
                   batch_norm_var_collection='moving_vars',
                   scope='vgg_16'):
    """This is a wrapper around the `_vgg_16_bn_relu`, which assumes that we
    want to run VGG-16 with batchnorm and RELU as a single network pose
    estimator (i.e. just the detector).
    """
    return _vgg_16_bn_relu(inputs=inputs,
                           num_classes=num_classes,
                           is_training=is_detector_training,
                           scope='vgg_16')


def vgg_bulat_cascade(inputs,
                      num_classes=16,
                      is_detector_training=True,
                      is_regressor_training=True,
                      scope='vgg_bulat_cascade'):
    """The cascade of VGG-style networks from the Bulat paper."""
    detect_logits, detect_endpoints = _vgg_16_bn_relu(inputs=inputs,
                                                      num_classes=num_classes,
                                                      is_training=is_detector_training,
                                                      scope='vgg_16')
    detect_endpoints['detect_logits'] = detect_logits

    stacked_heatmaps = tf.concat(values=[detect_logits, inputs], axis=3)

    regression_logits, _ = _vgg_bulat_regression(inputs=stacked_heatmaps,
                                                 num_classes=num_classes,
                                                 is_training=is_regressor_training,
                                                 scope=scope)

    return regression_logits, detect_endpoints


def vgg_bulat_cascade_maxpool_c2c3(inputs,
                                   num_classes=16,
                                   is_detector_training=True,
                                   is_regressor_training=True,
                                   scope='vgg_bulat_cascade'):
    """The cascade of VGG-style networks from the Bulat paper."""
    detect_logits, detect_endpoints = _vgg_16_bn_relu(inputs=inputs,
                                                      num_classes=num_classes,
                                                      is_training=is_detector_training,
                                                      scope='vgg_16')
    detect_endpoints['detect_logits'] = detect_logits

    stacked_heatmaps = tf.concat(values=[detect_logits, inputs], axis=3)

    regression_logits, _ = _vgg_bulat_regression_maxpool_c2c3(inputs=stacked_heatmaps,
                                                              num_classes=num_classes,
                                                              is_training=is_regressor_training,
                                                              scope=scope)

    return regression_logits, detect_endpoints


def vgg_bulat_cascade_maxpool_c3c4(inputs,
                                   num_classes=16,
                                   is_detector_training=True,
                                   is_regressor_training=True,
                                   scope='vgg_bulat_cascade'):
    """The cascade of VGG-style networks from the Bulat paper."""
    detect_logits, detect_endpoints = _vgg_16_bn_relu(inputs=inputs,
                                                      num_classes=num_classes,
                                                      is_training=is_detector_training,
                                                      scope='vgg_16')
    detect_endpoints['detect_logits'] = detect_logits

    stacked_heatmaps = tf.concat(values=[detect_logits, inputs], axis=3)

    regression_logits, _ = _vgg_bulat_regression_maxpool_c3c4(inputs=stacked_heatmaps,
                                                              num_classes=num_classes,
                                                              is_training=is_regressor_training,
                                                              scope=scope)

    return regression_logits, detect_endpoints


def vgg_bulat_cascade_maxpool_c2c3c4(inputs,
                                     num_classes=16,
                                     is_detector_training=True,
                                     is_regressor_training=True,
                                     scope='vgg_bulat_cascade'):
    """The cascade of VGG-style networks from the Bulat paper."""
    detect_logits, detect_endpoints = _vgg_16_bn_relu(inputs=inputs,
                                                      num_classes=num_classes,
                                                      is_training=is_detector_training,
                                                      scope='vgg_16')
    detect_endpoints['detect_logits'] = detect_logits

    stacked_heatmaps = tf.concat(values=[detect_logits, inputs], axis=3)

    regression_logits, _ = _vgg_bulat_regression_maxpool_c2c3c4(inputs=stacked_heatmaps,
                                                                num_classes=num_classes,
                                                                is_training=is_regressor_training,
                                                                scope=scope)

    return regression_logits, detect_endpoints


def vgg_bulat_cascade_conv3x3_c2c3c4(inputs,
                                     num_classes=16,
                                     is_detector_training=True,
                                     is_regressor_training=True,
                                     scope='vgg_bulat_cascade'):
    """The cascade of VGG-style networks from the Bulat paper."""
    detect_logits, detect_endpoints = _vgg_16_bn_relu(inputs=inputs,
                                                      num_classes=num_classes,
                                                      is_training=is_detector_training,
                                                      scope='vgg_16')
    detect_endpoints['detect_logits'] = detect_logits

    stacked_heatmaps = tf.concat(values=[detect_logits, inputs], axis=3)

    regression_logits, _ = _vgg_bulat_regression_conv3x3_c2c3c4(inputs=stacked_heatmaps,
                                                                num_classes=num_classes,
                                                                is_training=is_regressor_training,
                                                                scope=scope)

    return regression_logits, detect_endpoints


def two_vgg_16s_cascade(inputs,
                        num_classes=16,
                        is_detector_training=True,
                        is_regressor_training=True,
                        scope='vgg_bulat_cascade'):
    """A cascade of two VGG-16s, in place of the simpler VGG-style regression
    subnetwork from the Bulat paper.
    """
    detect_logits, detect_endpoints = _vgg_16_bn_relu(inputs=inputs,
                                                      num_classes=num_classes,
                                                      is_training=is_detector_training,
                                                      scope='vgg_16')
    detect_endpoints['detect_logits'] = detect_logits

    stacked_heatmaps = tf.concat(values=[detect_logits, inputs], axis=3)

    regression_logits, _ = _vgg_16_bn_relu(inputs=stacked_heatmaps,
                                           num_classes=num_classes,
                                           is_training=is_regressor_training,
                                           scope='vgg_16_regression')

    return regression_logits, detect_endpoints
