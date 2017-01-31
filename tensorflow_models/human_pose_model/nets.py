"""This module contains all of the model definitions, importing models from TF
Slim where needed.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import vgg
import bulat_vgg
import bulat_resnet

def vgg_loss(logits, endpoints, dense_joints, weights):
    """For VGG, currently we do a mean squared error on the joint locations
    compared with ground truth locations.
    """
    slim.losses.mean_squared_error(predictions=logits,
            labels=dense_joints,
            weights=weights)


def inception_v3_loss(logits, endpoints, dense_joints, weights):
    """The Inception architecture calculates loss on both the Auxiliary Logits
    and the final layer Logits.
    """
    auxiliary_logits = endpoints['AuxLogits']

    slim.losses.mean_squared_error(predictions=auxiliary_logits,
                                   labels=dense_joints,
                                   weights=weights,
                                   scope='aux_logits')

    slim.losses.mean_squared_error(predictions=logits,
                                   labels=dense_joints,
                                   weights=weights)

def detector_loss(logits, endpoints, heatmaps, weights):
    """Currently we regress joint gaussian confidence maps using pixel-wise L2 loss, based on
    Equation 2 of the paper.
    """
    slim.losses.sparse_softmax_cross_entropy(predictions=logits,
                                            labels=heatmaps,
                                            weights=weights,
                                            scope='detector_loss')

def regressor_loss(logits, endpoints, heatmaps, weights):
    """Currently we regress joint gaussian confidence maps using pixel-wise L2 loss, based on
    Equation 2 of the paper.
    """
    slim.losses.mean_squared_error(predictions=logits,
                                   labels=heatmaps,
                                   weights=weights,
                                   scope='regressor_loss')


NETS = {'vgg': vgg.vgg_16,
        'inception_v3': inception.inception_v3,
        'detector_vgg': bulat_vgg.detector_vgg,
        'regressor_vgg': bulat_vgg.regressor_vgg,
        'bulat_resnet': bulat_resnet.bulat_resnet_detector
        }

NET_ARG_SCOPES = {'vgg': vgg.vgg_arg_scope,
                  'inception_v3': inception.inception_v3_arg_scope,
                  'bulat_vgg': bulat_vgg.bulat_vgg_arg_scope,
                  'bulat_resnet': bulat_resnet.bulat_resnet_arg_scope}

NET_LOSS = {'vgg': vgg_loss,
            'inception_v3': inception_v3_loss,
            'detector_vgg': detector_loss,
            'regressor_vgg': regressor_loss,
            'bulat_resnet': regressor_loss }
