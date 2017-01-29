"""This module contains all of the model definitions, importing models from TF
Slim where needed.
"""
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import vgg
import vgg_bulat

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


def vgg_bulat_loss(logits, endpoints, heatmaps, weights):
    """Currently we regress joint heatmaps using pixel-wise L2 loss, based on
    Equation 2 of the paper.
    """
    slim.losses.mean_squared_error(predictions=logits,
                                   labels=heatmaps,
                                   weights=weights,
                                   scope='logits')


NETS = {'vgg': vgg.vgg_16,
        'inception_v3': inception.inception_v3,
        'vgg_bulat': vgg_bulat.vgg_16}
NET_ARG_SCOPES = {'vgg': vgg.vgg_arg_scope,
                  'inception_v3': inception.inception_v3_arg_scope,
                  'vgg_bulat': vgg_bulat.vgg_arg_scope}
NET_LOSS = {'vgg': vgg_loss,
            'inception_v3': inception_v3_loss,
            'vgg_bulat': vgg_bulat_loss}
