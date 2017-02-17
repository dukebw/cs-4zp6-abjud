"""This module contains all of the model definitions, importing models from TF
Slim where needed.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import vgg
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
                net_loss(logits, endpoints, heatmaps, weights)

            losses = tf.get_collection(key=tf.GraphKeys.LOSSES, scope=scope)
            regularization_losses = tf.get_collection(key=tf.GraphKeys.REGULARIZATION_LOSSES,
                                                      scope=scope)
            total_loss = tf.add_n(inputs=losses + regularization_losses, name='total_loss')
            total_loss = _summarize_loss(total_loss, gpu_index)

    return total_loss, logits


def vgg_loss(logits, endpoints, dense_joints, weights):
    """For VGG, currently we do a mean squared error on the joint locations
    compared with ground truth locations.
    """
    tf.losses.mean_squared_error(predictions=logits,
                                 labels=dense_joints,
                                 weights=weights)


def inception_v3_loss(logits, endpoints, dense_joints, weights):
    """The Inception architecture calculates loss on both the Auxiliary Logits
    and the final layer Logits.
    """
    auxiliary_logits = endpoints['AuxLogits']

    tf.losses.mean_squared_error(predictions=auxiliary_logits,
                                 labels=dense_joints,
                                 weights=weights,
                                 scope='aux_logits')

    tf.losses.mean_squared_error(predictions=logits,
                                 labels=dense_joints,
                                 weights=weights)


def sigmoid_cross_entropy_loss(logits, endpoints, binary_maps, weights):
    """
    Pixelwise cross entropy between binary masks and logits for each channel - see equation 1 in paper
    """
    tf.losses.sigmoid_cross_entropy(multi_class_labels=binary_maps,
                                    logits=logits,
                                    weights=weights,
                                    label_smoothing=0,
                                    scope='detector_loss')


def mean_squared_error_loss(logits, endpoints, heatmaps, weights):
    """Currently we regress joint gaussian confidence maps using pixel-wise L2 loss, based on
    Equation 2 of the paper.
    """
    tf.losses.mean_squared_error(predictions=logits,
                                 labels=heatmaps,
                                 weights=weights,
                                 scope='regressor_loss')


# Keeping the in for now for legacy
def vgg_bulat_loss(logits, endpoints, heatmaps, weights):
    """Currently we regress joint heatmaps using pixel-wise L2 loss, based on
    Equation 2 of the paper.
    """
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
