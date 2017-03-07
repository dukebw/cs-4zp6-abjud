# This code uses VGG architectures from vgg_bulat and builds a variational autoencoder
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

slim = tf.contrib.slim
add_arg_scope = tf.contrib.framework.add_arg_scope


def vgg_vae_arg_scope(weight_decay=0.0005):
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


def vgg_encode(inputs, batch_norm_params, scope, num_classes, dropout_keep_prob=0.5, is_training=True):
    """ Uses the base of the VGG-16 network (fully convolutional)
    for the encoder in the VAE
    """
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
                    a6 = slim.dropout(a6, dropout_keep_prob, is_training=is_training, scope='dropout6')
                    a7 = slim.conv2d(a6, 4096, [1, 1], scope='fc7')
                    a7 = slim.dropout(a7, dropout_keep_prob, is_training=is_training, scope='dropout7')

                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                    return a3, a4, a7, end_points


def vgg_decode(a3, a4, z, batch_norm_params, input_resolution, scope, num_classes):
    ''' Upsamples'''

    with tf.variable_scope(scope, 'vgg_16_decoder', [a3, a4, z]) as sc:
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

                    a8 = tf.image.resize_bilinear(images=z, size=a4.get_shape()[1:3])
                    skip_a4 = slim.conv2d(a4, num_classes, [1, 1],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          weights_initializer=tf.random_normal_initializer(),
                                          scope='skip_a4')
                    a8 = a8 + skip_a4

                    a8 = tf.image.resize_bilinear(images=a8, size=a3.get_shape()[1:3])
                    skip_a3 = slim.conv2d(a3, num_classes, [1, 1],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          weights_initializer=tf.random_normal_initializer(),
                                          scope='skip_a3')
                    a8 = a8 + skip_a3

                    a8 = tf.image.resize_bilinear(images=a8, size=input_resolution)

                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                    return a8, end_points


def sampleGaussian(mu, log_sigma):
    """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
    with tf.name_scope("sample_gaussian"):
        # reparameterization trick
        epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
        return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)


def _get_reparametrization(hidden_enc, num_classes):
    # latent distribution parameterized by hidden encoding
    # z ~ N(z_mean, np.exp(z_log_sigma)**2)
    z_mu = slim.conv2d(hidden_enc,
                         num_classes,
                         [1, 1],
                         activation_fn=None,
                         normalizer_fn=None,
                         scope='z_mu')

    z_log_sigma = slim.conv2d(hidden_enc,
                              num_classes,
                              [1, 1],
                              activation_fn=None,
                              normalizer_fn=None,
                              scope='z_log_sigma')

    return sampleGaussian(z_mu, z_log_sigma), z_mu, z_log_sigma


def vgg_vae_v0(inputs,
              num_classes=16,
              is_detector_training=True,
              is_regressor_training=False,
              dropout_keep_prob=0.5,
              batch_norm_var_collection='moving_vars',
              scope='vgg_16'):

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
      'is_training': is_detector_training
    }

    a3, a4, hidden_enc, end_points = vgg_encode(inputs,
                                               batch_norm_params,
                                               scope,
                                               num_classes,
                                               dropout_keep_prob,
                                               is_detector_training)

    z, z_mu, z_log_sigma = _get_reparametrization(hidden_enc, num_classes)

    end_points['z_mu'] = z_mu
    end_points['z_log_sigma'] = z_log_sigma

    a8,_ = vgg_decode(a3, a4, z, batch_norm_params, inputs.get_shape()[1:3], scope, num_classes)

    return a8, end_points
