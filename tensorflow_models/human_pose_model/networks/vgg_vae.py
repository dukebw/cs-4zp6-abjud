# This code uses VGG architectures from vgg_bulat and builds a variational autoencoder

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

slim = tf.contrib.slim

import vgg_bulat


def encode(inputs):
    return vgg_bulat._vgg_16_bn_relu(inputs)


def decode(a3, a4, a8):
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


def sampleGaussian(mu, log_sigma):
    """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
    with tf.name_scope("sample_gaussian"):
        # reparameterization trick
        epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
        return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)


def reparametrize(hidden_representation)
    # latent distribution parameterized by hidden encoding
    # z ~ N(z_mean, np.exp(z_log_sigma)**2)
    z_mean = slim.fully_connected(hidden_representation, num_latent_dim)
    z_log_sigma = slim.fully_connected(hidden_representation, num_latent_dim)

    # kingma & welling: only 1 draw necessary as long as minibatch large enough (>100)
    return sampleGaussian(z_mean, z_log_sigma)


def vae(inputs,
        num_classes=16,
        is_training=True,
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
                    hidden_var = encode(inputs)
                    z = reparametrize(hidden_var)
                    decode(z)

