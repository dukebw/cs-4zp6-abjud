# This code uses VGG architectures from vgg_bulat and builds a variational autoencoder
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

slim = tf.contrib.slim
add_arg_scope = tf.contrib.framework.add_arg_scope
NUM_CLASSES = 16
LATENT_DIM = 16


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


def vgg_16_base(inputs, dropout_keep_prob, is_training):
    """Base of the VGG-16 network (fully convolutional).

    Note that this function only constructs the network, and needs to be called
    with the correct variable scopes and tf.slim arg scopes.
    """
    a1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    print(a1.get_shape())
    a1 = slim.max_pool2d(a1, [2, 2], scope='pool1')
    print(a1.get_shape())
    a2 = slim.repeat(a1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    print(a2.get_shape())
    a2 = slim.max_pool2d(a2, [2, 2], scope='pool2')
    print(a2.get_shape())
    a3 = slim.repeat(a2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    print(a2.get_shape())
    a3 = slim.max_pool2d(a3, [2, 2], scope='pool3')
    print(a3.get_shape())
    a4 = slim.repeat(a3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    print(a3.get_shape())
    a4 = slim.max_pool2d(a4, [2, 2], scope='pool4')
    print(a4.get_shape())
    a5 = slim.repeat(a4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    print(a5.get_shape())
    a5 = slim.max_pool2d(a5, [2, 2], scope='pool5')
    print(a5.get_shape())

    # Use conv2d instead of fully_connected layers.
    a6 = slim.conv2d(a5, 4096, [7, 7], scope='fc6')
    print(a5.get_shape())
    a6 = slim.dropout(a6, dropout_keep_prob, is_training=is_training, scope='dropout6')
    print(a6.get_shape())
    a7 = slim.conv2d(a6, 4096, [1, 1], scope='fc7')
    print(a6.get_shape())
    a7 = slim.dropout(a7, dropout_keep_prob, is_training=is_training, scope='dropout7')
    print(a7.get_shape())

    a8  = slim.conv2d(a7, NUM_CLASSES, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
    print(a8.get_shape())

    return a3, a4, a8


def _get_pose_logits(a3,a4,a8,resolution):
    a9 = tf.image.resize_bilinear(images=a8, size=a4.get_shape()[1:3])
    skip_a4 = slim.conv2d(a4, NUM_CLASSES, [1, 1], scope='skip_a4')
    a9 = a9 + skip_a4

    a9 = tf.image.resize_bilinear(images=a9, size=a3.get_shape()[1:3])
    skip_a3 = slim.conv2d(a3, NUM_CLASSES, [1, 1], scope='skip_a3')
    a9 = a9 + skip_a3

    a9 = tf.image.resize_bilinear(images=a9, size=resolution)
    return a9


def sampleGaussian(mu, log_sigma):
    """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
    with tf.name_scope("sample_gaussian"):
        # reparameterization trick
        epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
        return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)

def _get_reparameterization(hidden_enc):
    # latent distribution parameterized by hidden encoding
    # z ~ N(z_mean, np.exp(z_log_sigma)**2)
    z_mu = slim.fully_connected(hidden_enc,
                                LATENT_DIM,
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='z_mu')

    z_log_sigma = slim.fully_connected(hidden_enc,
                                       LATENT_DIM,
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       scope='z_log_sigma')

    return sampleGaussian(z_mu, z_log_sigma), z_mu, z_log_sigma


def faster_pixel_cnn(img):
    if step == 0:
        pass
        # Prediction 1

        # Prediction 2

        # Prediction 3

        # Prediction 4
        # Step function - should be with a while loop, cond, body, var like in raw_rnn


def _get_img_reconstruction(a8,z,batch_norm_params):
    shape = z.get_shape().as_list()
    size = shape[1]*shape[1]
    d1 = slim.fully_connected(z,
                              size,
                              activation_fn=tf.nn.relu,
                              normalizer_fn=slim.batch_norm,
                              normalizer_params=batch_norm_params)

    d1 = tf.reshape(d1, [9, LATENT_DIM, LATENT_DIM,1])
    d2 = slim.conv2d_transpose(d1,1,2,2)
    d3 = slim.conv2d_transpose(d2,1,2,2)
    img = slim.conv2d_transpose(d3,3,2,2)
    return img


def _vgg_16_vae_v0(inputs,
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
      NUM_CLASSES: number of predicted classes (joints).
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
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.conv2d_transpose, slim.fully_connected, slim.flatten],
                                outputs_collections=end_points_collection):
                # nested argscope because we don't apply activations and normalization to maxpool
                with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params=batch_norm_params):

                    a3, a4, a8 = vgg_16_base(inputs,
                                             dropout_keep_prob,
                                             is_training)

                    z = slim.flatten(a8)
                    z_latent, z_mu, z_log_sigma = _get_reparameterization(z)

                    pose_logits = _get_pose_logits(a3,a4,a8, inputs.get_shape()[1:3])
                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                    # For calculating the KL loss
                    end_points['z_mu'] = z_mu
                    end_points['z_log_sigma'] = z_log_sigma
                    end_points['pose_logits'] = pose_logits

                    return pose_logits, end_points


def vgg_16_vae_v0(inputs,
                   is_detector_training=True,
                   is_regressor_training=True,
                   dropout_keep_prob=0.5,
                   batch_norm_var_collection='moving_vars',
                   scope='vgg_16'):
    """This is a wrapper around the `_vgg_16_bn_relu`, which assumes that we
    want to run VGG-16 with batchnorm and RELU as a single network pose
    estimator (i.e. just the detector).
    """
    return _vgg_16_vae_v0(inputs=inputs,
                           is_training=is_detector_training,
                           scope='vgg_16')
