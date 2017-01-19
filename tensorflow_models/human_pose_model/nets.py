"""This module contains all of the model definitions, importing models from TF
Slim where needed.
"""
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import vgg

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


NETS = {'vgg': vgg.vgg_16,
        'inception_v3': inception.inception_v3}
NET_ARG_SCOPES = {'vgg': vgg.vgg_arg_scope,
                  'inception_v3': inception.inception_v3_arg_scope}
NET_LOSS = {'vgg': vgg_loss,
            'inception_v3': inception_v3_loss}
