"""This module contains all of the model definitions, importing models from TF
Slim where needed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import vgg

import tensorflow as tf
import tensorflow.contrib.slim as slim
import vgg_bulat
import resnet_bulat

# TODO(brendan): separate file
NUM_JOINTS = 16

def _summarize_loss(total_loss, gpu_index):
    """Summarizes the loss and average loss for this tower, and ensures that
    loss averages are computed every time the loss is computed.
    """
    loss_averages = tf.train.ExponentialMovingAverage(decay=0.9, name='avg')
    loss_averages_op = loss_averages.apply(var_list=[total_loss])
    with tf.control_dependencies(control_inputs=[loss_averages_op]):
        total_loss = tf.identity(total_loss)

    return total_loss

def _scale_losses(losses, weights):
  	"""Computes the scaled loss.
  	Args:
  	  losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
  	  weights: `Tensor` of shape `[]`, `[batch_size]` or
  	    `[batch_size, d1, ... dN]`. The `losses` are reduced (`tf.reduce_sum`)
  	    until its dimension matches that of `weights` at which point the reduced
  	    `losses` are element-wise multiplied by `weights` and a final `reduce_sum`
  	    is computed on the result. Conceptually, this operation is similar to
  	    broadcasting (tiling) `weights` to be the same shape as `losses`,
  	    performing an element-wise multiplication, and summing the result. Note,
  	    however, that the dimension matching is right-to-left, not left-to-right;
  	    i.e., the opposite of standard NumPy/Tensorflow broadcasting.
  	Returns:
  	  A scalar tf.float32 `Tensor` whose value represents the sum of the scaled
  	    `losses`.
  	"""
  	weighted_losses = math_ops.multiply(losses, weights)
  	return math_ops.reduce_sum(weighted_losses)


def _safe_div(numerator, denominator, name="value"):
  	"""Computes a safe divide which returns 0 if the denominator is zero.
  	Note that the function contains an additional conditional check that is
  	necessary for avoiding situations where the loss is zero causing NaNs to
 	 creep into the gradient computation.

  	Args:
  	  numerator: An arbitrary `Tensor`.
  	  denominator: `Tensor` whose shape matches `numerator` and whose values are
  	    assumed to be non-negative.
  	  name: An optional name for the returned op.
  	Returns:
  	  The element-wise value of the numerator divided by the denominator.
  	"""
  	return array_ops.where(math_ops.greater(denominator, 0),
  	    				   math_ops.div(numerator,
						   array_ops.where(math_ops.equal(denominator, 0),
  	        			   array_ops.ones_like(denominator),
						   denominator)),
  	    				   array_ops.zeros_like(numerator),
  	    				   name=name)


def _safe_mean(losses, num_present):
    """Computes a safe mean of the losses.
    Args:
        losses: `Tensor` whose elements contain individual loss measurements.
        num_present: The number of measurable elements in `losses`.
    Returns:
        A scalar representing the mean of `losses`. If `num_present` is zero,
        then zero is returned.
    """
    total_loss = math_ops.reduce_sum(losses)
    return _safe_div(total_loss, num_present)


def _num_present(losses, weights, per_batch=False):
    """Computes the number of elements in the loss function induced by `weights`.
    A given weights tensor induces different numbers of usable elements in the
    `losses` tensor. The `weights` tensor is broadcast across `losses` for all
    possible dimensions. For example, if `losses` is a tensor of dimension
    `[4, 5, 6, 3]` and `weights` is a tensor of shape `[4, 5]`, then `weights` is,
    in effect, tiled to match the shape of `losses`. Following this effective
    tile, the total number of present elements is the number of non-zero weights.
    Args:
      losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
      weights: `Tensor` of shape `[]`, `[batch_size]` or
        `[batch_size, d1, ... dK]`, where K < N.
      per_batch: Whether to return the number of elements per batch or as a sum
        total.
    Returns:
      The number of present (non-zero) elements in the losses tensor. If
        `per_batch` is `True`, the value is returned as a tensor of size
        `[batch_size]`. Otherwise, a single scalar tensor is returned.
    """
    with ops.name_scope(None, "num_present", (losses, weights)) as scope:
        weights = math_ops.to_float(weights)
        present = array_ops.where(math_ops.equal(weights, 0.0), array_ops.zeros_like(weights), array_ops.ones_like(weights))
		present = weights_broadcast_ops.broadcast_weights(present, losses)
        if per_batch:
            return math_ops.reduce_sum(present, axis=math_ops.range(1, array_ops.rank(present)), keep_dims=True, name=scope)
        return math_ops.reduce_sum(present, name=scope)


def compute_weighted_loss(losses, 
						  weights,
						  scope=None, 
						  loss_collection=ops.GraphKeys.LOSSES):
  	"""Computes the weighted loss.
  	Args:
    	losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    	weights: Optional `Tensor` whose rank is either 0, or the same rank as
    	  `losses`, and must be broadcastable to `losses` (i.e., all dimensions must
    	  be either `1`, or the same as the corresponding `losses` dimension).
    	scope: the scope for the operations performed in computing the loss.
    	loss_collection: the loss will be added to these collections.
  	Returns:
  	  A scalar `Tensor` that returns the weighted loss.
  	Raises:
  	  ValueError: If `weights` is `None` or the shape is not compatible with
  	    `losses`, or if the number of dimensions (rank) of either `losses` or
  	    `weights` is missing.
  	"""
	# We broadcast down via reduce_sum per image
	weights = tf.concat(tf.ones(shape = losses.get_shape()[0], weights)
  	with ops.name_scope(scope, "weighted_loss", (losses, weights)):
  	  	with ops.control_dependencies((weights_broadcast_ops.assert_broadcastable(weights, losses),)):
			losses = ops.convert_to_tensor(losses)
  	  	  	input_dtype = losses.dtype
  	  	  	losses = math_ops.to_float(losses)
  	  	  	weights = math_ops.to_float(weights)
  	  	  	total_loss = _scale_losses(losses, weights)
  	  	  	num_present = _num_present(losses, weights)
  	  	  	mean_loss = _safe_mean(total_loss, num_present)
  	  	  	# Convert the result back to the input type.
  	  	  	mean_loss = math_ops.cast(mean_loss, input_dtype)
  	  	  	util.add_loss(mean_loss, loss_collection)
			return mean_loss


def inference(images,
              binary_maps,
              heatmaps,
              weights,
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
                                                    num_classes=NUM_JOINTS,
                                                    is_training=is_training,
                                                    scope=scope)
                net_loss(logits, endpoints, heatmaps, binary_maps, weights)

            losses = tf.get_collection(key=tf.GraphKeys.LOSSES, scope=scope)
            regularization_losses = tf.get_collection(key=tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
            total_loss = tf.add_n(inputs=losses + regularization_losses + user_loss, name='total_loss')
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


def sigmoid_cross_entropy_loss(logits, endpoints, heatmaps, binary_maps, weights):
    """Pixelwise cross entropy between binary masks and logits for each channel.

    See equation 1 in Bulat paper.
    """
    """ Does not work because it reduces the mean across all dimensions
    tf.losses.sigmoid_cross_entropy(multi_class_labels=binary_maps,
                                    logits=logits,
                                    weights=weights,
                                    label_smoothing=0,
                                    scope='detector_loss')
    """
    # componentwise logistic losses
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=binary_maps,
                                                     logits=logits,
                                                     name='cross_entropy_bulat')

	return _compute_weighted_loss(losses, weights, scope='cross_entropy_bulat')

def mean_squared_error_loss(logits, endpoints, heatmaps, binary_maps, weights):
    """Currently we regress joint gaussian confidence maps using pixel-wise L2 loss, based on
    Equation 2 of the paper.
    """
    tf.losses.mean_squared_error(predictions=logits,
                                 labels=heatmaps,
                                 weights=weights,
                                 scope='regressor_loss')


# Keeping the in for now for legacy
#
def vgg_bulat_loss(logits, endpoints, heatmaps, binary_maps, weights):
    """Currently we regress joint heatmaps using pixel-wise L2 loss, based on
    Equation 2 of the paper.
    """
    tf.losses.sigmoid_cross_entropy(multi_class_labels=binary_maps,
                                    logits=logits,
                                    weights=weights,
                                    label_smoothing=0,
                                    scope='detector_loss')

    tf.losses.mean_squared_error(predictions=logits,
                                 labels=heatmaps,
                                 weights=weights,
                                 scope='mean_squared_loss')


NETS = {'vgg': (vgg.vgg_16, vgg.vgg_arg_scope),
        'inception_v3': (inception.inception_v3, inception.inception_v3_arg_scope),
        'vgg_bulat': (vgg_bulat.vgg_16, vgg_bulat.vgg_arg_scope),
        'vgg_fcn32': (vgg_bulat.vgg_16_fcn32, vgg_bulat.vgg_arg_scope),
        'vgg_bulat_bn_relu': (vgg_bulat.vgg_16_bn_relu, vgg_bulat.vgg_arg_scope),
        'resnet_bulat': (resnet_bulat.resnet_detector, resnet_bulat.resnet_arg_scope)}

NET_LOSS = {'sigmoid_cross_entropy_loss': sigmoid_cross_entropy_loss,
            'mean_squared_error_loss': mean_squared_error_loss,
            'inception_v3_loss': inception_v3_loss}
