"""This module contains all of the model definitions, importing models from TF
Slim where needed.
"""

from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import vgg

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import tensorflow.contrib.slim as slim
from dataset.mpii_datatypes import Person
from networks import vgg_bulat
from networks import resnet_bulat

def _summarize_loss(total_loss, gpu_index):
    """Summarizes the loss and average loss for this tower, and ensures that
    loss averages are computed every time the loss is computed.
    """
    loss_averages = tf.train.ExponentialMovingAverage(decay=0.9, name='avg')
    loss_averages_op = loss_averages.apply(var_list=[total_loss])
    with tf.control_dependencies(control_inputs=[loss_averages_op]):
        total_loss = tf.identity(total_loss)

    return total_loss


def inference(images,
              binary_maps,
              heatmaps,
              weights,
              is_visible_weights,
              gpu_index,
              network_name,
              loss_name,
              is_training,
              scope):
    """Sets up a human pose inference model, computes predictions on input
    images and calculates loss on those predictions based on an input dense
    vector of joint location confidence maps and binary maps (the ground truth
    vector).

    TF-slim's `arg_scope` is used to keep variables (`slim.model_variable`) in
    CPU memory. See the training procedure block diagram in the TF Inception
    [README](https://github.com/tensorflow/models/tree/master/inception).

    Args:
        images: Mini-batch of preprocessed examples dequeued from the input
            pipeline.
        heatmaps: Confidence maps of ground truth joints.
        weights: Weights of heatmaps (tensors of all 1s if joint present, all
            0s if not present).
        is_visible_weights: Weights of heatmaps/binary maps, with occluded
            joints zero'ed out.
        gpu_index: Index of GPU calculating the current loss.
        scope: Name scope for ops, which is different for each tower (tower_N).

    Returns:
        Tensor giving the total loss (combined loss from auxiliary and primary
        logits, added to regularization losses).
    """
    part_detect_net = NETS[network_name][0]
    net_arg_scope = NETS[network_name][1]
    net_loss = NET_LOSS[loss_name]
    with slim.arg_scope([slim.model_variable], device='/cpu:0'):
        with slim.arg_scope(net_arg_scope()):
            with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=(gpu_index > 0)):
                logits, endpoints = part_detect_net(inputs=images,
                                                    num_classes=Person.NUM_JOINTS,
                                                    is_training=is_training,
                                                    scope=scope)
                net_loss(logits, endpoints, heatmaps, binary_maps, weights, is_visible_weights)

            losses = tf.get_collection(key=tf.GraphKeys.LOSSES, scope=scope)

            regularization_losses = tf.get_collection(
                key=tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)

            total_loss = tf.add_n(inputs=losses + regularization_losses,
                                  name='total_loss')
            total_loss = _summarize_loss(total_loss, gpu_index)

    return total_loss, logits


def inception_v3_loss(logits, endpoints, dense_joints, weights):
    """The Inception architecture calculates loss on both the Auxiliary Logits
    and the final layer Logits.

    TODO(brendan): correct this to work with our confidence map/binary map
    labels.
    """
    auxiliary_logits = endpoints['AuxLogits']

    tf.losses.mean_squared_error(predictions=auxiliary_logits,
                                 labels=dense_joints,
                                 weights=weights,
                                 scope='aux_logits')

    tf.losses.mean_squared_error(predictions=logits,
                                 labels=dense_joints,
                                 weights=weights)


def _add_weighted_loss_to_collection(losses, weights):
    """Weights `losses` by weights, and adds the weighted losses, normalized by
    the number of joints present, to `tf.GraphKeys.LOSSES`.

    Specifically, the losses are summed across all dimensions (x, y,
    num_joints), producing a scalar loss per batch. That scalar loss then needs
    to be normalized by the number of joints present. This is equivalent to
    sum(weights[:, 0, 0, :]), since `weights` is a [image_dim, image_dim] map
    of eithers all 1s or all 0s, depending on whether a joints is present or
    not, respectively.

    Args:
        losses: Element-wise losses as calculated by your favourite function.
        weights: Element-wise weights.
    """
    losses = tf.transpose(a=losses, perm=[1, 2, 0, 3])
    weighted_loss = tf.multiply(losses, weights)
    per_batch_loss = tf.reduce_sum(input_tensor=weighted_loss, axis=[0, 1, 3])

    num_joints_present = tf.reduce_sum(input_tensor=weights, axis=1)

    assert_safe_div = tf.assert_greater(num_joints_present, 0.0)
    with tf.control_dependencies(control_inputs=[assert_safe_div]):
        per_batch_loss /= num_joints_present

    total_loss = tf.reduce_mean(input_tensor=per_batch_loss)
    tf.add_to_collection(name=tf.GraphKeys.LOSSES, value=total_loss)


def _sigmoid_cross_entropy_loss(logits, binary_maps, weights):
    """Pixelwise cross entropy between binary masks and logits for each channel.

    See equation 1 in Bulat paper.
    """
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=binary_maps,
                                                     logits=logits,
                                                     name='cross_entropy_bulat')

    _add_weighted_loss_to_collection(losses, weights)


def _mean_squared_error_loss(logits, heatmaps, weights):
    """Currently we regress joint gaussian confidence maps using pixel-wise L2
    loss, based on Equation 2 of the paper.
    """
    losses = tf.square(tf.subtract(logits, heatmaps))
    _add_weighted_loss_to_collection(losses, weights)


def detector_only_xentropy_loss(logits,
                                endpoints,
                                heatmaps,
                                binary_maps,
                                weights,
                                is_visible_weights):
    """Trains only the detector, using pixel-wise sigmoid cross-entropy loss.

    Trains only on visible joints.
    """
    _sigmoid_cross_entropy_loss(logits, binary_maps, is_visible_weights)


def detector_only_regression_loss(logits,
                                  endpoints,
                                  heatmaps,
                                  binary_maps,
                                  weights,
                                  is_visible_weights):
    """Trains only the detector, using regression (pixel-wise L2 loss)."""
    _mean_squared_error_loss(logits, heatmaps, weights)


def both_nets_xentropy_regression_loss(logits,
                                       endpoints,
                                       heatmaps,
                                       binary_maps,
                                       weights,
                                       is_visible_weights):
    """Currently we regress joint heatmaps using pixel-wise L2 loss, based on
    Equation 2 of the paper.

    Cross-entropy on visible joints, with endpoints from first network.
    Regression on outputs from second network.
    """
    _sigmoid_cross_entropy_loss(endpoints['detect_logits'],
                                binary_maps,
                                is_visible_weights)

    _mean_squared_error_loss(logits, heatmaps, weights)


def both_nets_regression_loss(logits,
                              endpoints,
                              heatmaps,
                              binary_maps,
                              weights,
                              is_visible_weights):
    """Currently we regress joint heatmaps using pixel-wise L2 loss, based on
    Equation 2 of the paper.

    Regression on outputs from each network.
    """
    _mean_squared_error_loss(endpoints['detect_logits'],
                             heatmaps,
                             is_visible_weights)

    _mean_squared_error_loss(logits, heatmaps, weights)


NETS = {'vgg': (vgg.vgg_16, vgg.vgg_arg_scope),
        'inception_v3': (inception.inception_v3, inception.inception_v3_arg_scope),
        'vgg_bulat_cascade': (vgg_bulat.vgg_bulat_cascade, vgg_bulat.vgg_arg_scope),
        'vgg_fcn32': (vgg_bulat.vgg_16_fcn32, vgg_bulat.vgg_arg_scope),
        'vgg_bulat_bn_relu': (vgg_bulat.vgg_16_bn_relu, vgg_bulat.vgg_arg_scope),
        'resnet_bulat': (resnet_bulat.resnet_detector, resnet_bulat.resnet_arg_scope)}

NET_LOSS = {'detector_only_regression': detector_only_regression_loss,
            'detector_only_xentropy': detector_only_xentropy_loss,
            'both_nets_regression': both_nets_regression_loss,
            'both_nets_xentropy_regression': both_nets_xentropy_regression_loss,
            'inception_v3_loss': inception_v3_loss}
