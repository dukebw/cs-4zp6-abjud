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
"""Contains definitions for the preactivation form of Residual Networks.

The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v1

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf

from networks import resnet_utils

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope

PYRAMID_LEVELS = {
    'resnet_v1_50_pyramid': [(4, 'block3/unit_5/bottleneck_v1', 1024),
                             (3, 'block2/unit_3/bottleneck_v1', 512),
                             (2, 'block1/unit_2/bottleneck_v1', 256)],
    'resnet_v1_152_pyramid': [(4, 'block3/unit_35/bottleneck_v1', 1024),
                              (3, 'block2/unit_7/bottleneck_v1', 512),
                              (2, 'block1/unit_2/bottleneck_v1', 256)]
}

@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
    """Bottleneck residual unit variant with BN after convolutions.

    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that we use here the bottleneck variant which has an
    extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.

    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride,
                                   activation_fn=None, scope='shortcut')

        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               activation_fn=None, scope='conv3')

        output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)


def _deconv_bilinear_upsample_initialized(inputs, stride, out_channels):
    """Creates a deconvolution layers that is initialized to bilinear
    upsampling, with no batch norm and no activation function.

    Based on the bilinear interpolation examples
    [here](http://avisynth.nl/index.php/Resampling)
    and [here]
    (http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/).
    [This guide](https://arxiv.org/pdf/1603.07285.pdf) is also a visual
    reference guide for deconvolutions.

    Args:
        inputs: The inputs to the deconvolution layer.
        stride: Factor by which to upsample.
        out_channels: Channels of the output. Currently it is assumed that the
                      inputs are of the same number of channels.

    Returns:
        Deconvolution layer initialized to bilinear upsampling.
    """
    kernel_size = 2*stride - (stride % 2)
    center = (kernel_size - 1)/2.0

    ogrid = np.ogrid[:kernel_size, :kernel_size]
    bilinear = ((1 - abs(ogrid[0] - center)/stride)*
                (1 - abs(ogrid[1] - center)/stride))

    input_shape = tf.shape(inputs)
    output_shape = [input_shape[0],
                    stride*input_shape[1],
                    stride*input_shape[2],
                    out_channels]

    weights = np.zeros([kernel_size, kernel_size, out_channels, out_channels],
                       dtype=np.float32)
    for i in range(out_channels):
        weights[:, :, i, i] = bilinear

    return slim.conv2d_transpose(inputs=inputs,
                                 num_outputs=out_channels,
                                 kernel_size=[kernel_size, kernel_size],
                                 stride=stride,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 weights_initializer=tf.constant_initializer(bilinear),
                                 scope='upsample')


def _top_down_lateral_merge(top_down_net,
                            scope_name,
                            end_points,
                            lateral_net_name,
                            lateral_net_channels):
    """Merges the `top_down_net` activations with the activations corresponding
    to `lateral_net_name` using the building block of Figure 3 from
    (Feaure Pyramid Networks for Object Detection)
    [https://arxiv.org/pdf/1612.03144.pdf].

    The building block consists of deconvolving `top_down_net` with stride of
    two (and also reducing to the same number of channels as the lateral net),
    running the lateral net through a 1x1 convolution, adding these two
    activations and running the result through a 3x3 convolution, which is said
    in the paper to "prevent aliasing".

    Args:
        top_down_net: The spatially coarser, semantically stronger feature maps
            from higher pyramid levels.
        scope_name: Original scope name.
        end_points: Named activations in the network.
        lateral_net_name: Name of the lateral network (higher resolution
            activations). Needed to retrieve the activations from `end_points`.
        lateral_net_channels: Number of channels of lateral net.

    Returns:
        Merged top-down pathway activations and lateral connections.
    """
    lateral_net = end_points[scope_name + lateral_net_name]
    assert_shape_equal = tf.assert_equal(lateral_net_channels,
                                         tf.shape(lateral_net)[3])
    with tf.control_dependencies(control_inputs=[assert_shape_equal]):
        top_down_net = slim.conv2d_transpose(inputs=top_down_net,
                                             num_outputs=lateral_net_channels,
                                             kernel_size=[4, 4],
                                             stride=2,
                                             scope='deconv')

        lateral_net = slim.conv2d(lateral_net,
                                  lateral_net_channels,
                                  [1, 1],
                                  scope='lateral')

        return slim.conv2d(top_down_net + lateral_net,
                           lateral_net_channels,
                           [3, 3],
                           scope='merge')


def _merge_with_detect_logits(top_down_net, detect_logits):
    """Downsamples `detect_logits` to the size of `top_down_net` and returns
    the concatenation of the result, or just `top_down_net` if `detect_logits`
    is None.
    """
    if detect_logits is None:
        return top_down_net

    merge_detect_logits = tf.image.resize_bilinear(
        detect_logits, tf.shape(top_down_net)[1:3])
    return tf.concat([top_down_net, merge_detect_logits], axis=3)


def _compute_pyramid_prediction(level_features,
                                detect_logits,
                                num_classes,
                                scope):
    """Computes predictions for one level of the feature pyramid network."""
    top_down_net = _merge_with_detect_logits(level_features, detect_logits)
    level_logits = slim.conv2d(level_features,
                               num_classes,
                               [1, 1],
                               activation_fn=None,
                               normalizer_fn=None,
                               scope=scope)

    return top_down_net, level_logits


def _construct_pyramid(c5,
                       end_points,
                       detect_logits,
                       num_classes,
                       scope_name,
                       pyramid_name):
    """Constructs an entire pyramid head from the FPN paper (Lin et al.),
    starting with `c5` as the activations from the top-down pathway.

    For each resolution level, the activations from the layer immediately
    before the downsample to the next resolution level are merged with
    upsampled activations from the top-down (semantically strong) pathway.
    After each of the merges of lateral and top-down pathways, predictions are
    made, in the sense that these raw logits are added to the overall logits
    prediction of human joint positions.

    Args:
        c5: Layer immediately before making predictions from the topmost level.
        end_points: End point activations in the network.
        detect_logits: Logits from the detection head, or `None`.
        num_classes: Number of joints.
        scope_name: Based name of scope, used to get activations from
            end_points.
        pyramid_name: Name to add to end of scope for this pyramid head.
    """
    pyramid_levels = PYRAMID_LEVELS[pyramid_name]
    with tf.variable_scope(pyramid_name):
        top_down_net, p5 = _compute_pyramid_prediction(c5,
                                                       detect_logits,
                                                       num_classes,
                                                       'p5/logits')

        upsample_factor = 32
        logits = _deconv_bilinear_upsample_initialized(p5,
                                                       upsample_factor,
                                                       num_classes)

        for pyramid in pyramid_levels:
            with tf.variable_scope("p{}".format(pyramid[0])):
                next_features = _top_down_lateral_merge(top_down_net,
                                                        scope_name,
                                                        end_points,
                                                        pyramid[1],
                                                        pyramid[2])

                top_down_net, next_logits = _compute_pyramid_prediction(
                    next_features, detect_logits, num_classes, 'logits')

                upsample_factor = int(upsample_factor/2)
                logits += _deconv_bilinear_upsample_initialized(next_logits,
                                                                upsample_factor,
                                                                num_classes)

        return logits


def bulat_resnet_v1(inputs,
                    blocks,
                    num_classes=None,
                    is_training=True,
                    output_stride=None,
                    scope=None):
    """
    "Blocks B1-B4 are the same as the ones in the original ResNet, and B5 was
    slightly modified. We firstly removed both the fully connected layer after B5
    and then the preceding average pooling layer. Then, we added a scoring
    convolutional layer B6 with N outputs, one for each part. Next, to address
    the extremely low output resolution, we firstly modified B5 by changing the
    stride of its convolutional layers from 2px to 1px and then added (after B6)
    a deconvolution layer B7 with a kernel size and stride of 4, that upsamples
    the output layers to match the resolution
    of the input."
    """
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=None) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d,
                             slim.conv2d_transpose,
                             bottleneck,
                             resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs

                net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

                c5 = resnet_utils.stack_blocks_dense(net, blocks, output_stride)

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                logits = _construct_pyramid(c5,
                                            end_points,
                                            None,
                                            num_classes,
                                            sc.original_name_scope,
                                            scope + '_pyramid')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                return logits, end_points


def resnet_50_detector(inputs,
                       num_classes=16,
                       is_detector_training=True,
                       is_regressor_training=True,
                       scope='resnet_v1_50'):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""

    blocks = [
            resnet_utils.Block('block1', bottleneck,
                [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            resnet_utils.Block('block2', bottleneck,
                [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            resnet_utils.Block('block3', bottleneck,
                [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
            resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]

    return bulat_resnet_v1(inputs,
                           blocks,
                           num_classes,
                           is_training=is_detector_training,
                           scope='resnet_v1_50')


def resnet_152_two_heads_parallel(inputs,
                                 num_classes=16,
                                 is_detector_training=True,
                                 is_regressor_training=True,
                                 scope='resnet_152_two_heads'):
    """Runs a ResNet-152 with a regression and detection head in parallel."""
    return _resnet_two_heads_inner(resnet_152_detector,
                                   True,
                                   inputs,
                                   num_classes=16,
                                   is_detector_training=True,
                                   is_regressor_training=True,
                                   shared_net_scope_name='resnet_v1_152')


def resnet_50_two_heads_parallel(inputs,
                                 num_classes=16,
                                 is_detector_training=True,
                                 is_regressor_training=True,
                                 scope='resnet_50_two_heads'):
    """Runs a ResNet-50 with a regression and detection head in parallel."""
    return _resnet_two_heads_inner(resnet_50_detector,
                                   True,
                                   inputs,
                                   num_classes=16,
                                   is_detector_training=True,
                                   is_regressor_training=True,
                                   shared_net_scope_name='resnet_v1_50')


def resnet_50_two_heads_serial(inputs,
                               num_classes=16,
                               is_detector_training=True,
                               is_regressor_training=True,
                               scope='resnet_50_two_heads'):
    """Runs a ResNet-50 with a regression and detection head in serial."""
    return _resnet_two_heads_inner(resnet_50_detector,
                                   False,
                                   inputs,
                                   num_classes=16,
                                   is_detector_training=True,
                                   is_regressor_training=True,
                                   shared_net_scope_name='resnet_v1_50')


def _resnet_two_heads_inner(resnet_detector,
                            parallelize_heads,
                            inputs,
                            num_classes=16,
                            is_detector_training=True,
                            is_regressor_training=True,
                            shared_net_scope_name='resnet_v1_50'):
    """Creates one core ResNet with two heads: one for detection (cross-entropy
    loss on visible joints) and one for regression (L2 loss on all labelled
    joints).

    Depending on `parallelize_heads`, the heads are either run in parallel, or
    in serial, where serial means that the binary maps output by the detector
    are concatenated with the inputs to the top-down/lateral merge building
    blocks of the regression head.
    """
    detect_logits, detect_end_points = resnet_detector(inputs,
                                                       num_classes=num_classes,
                                                       is_detector_training=is_detector_training,
                                                       is_regressor_training=is_regressor_training,
                                                       scope=shared_net_scope_name)
    detect_end_points['detect_logits'] = detect_logits

    original_scope_name = (detect_logits.op.name.split('/')[0] +
                           '/' +
                           shared_net_scope_name +
                           '/')
    c5 = detect_end_points[shared_net_scope_name + '/block4']
    with slim.arg_scope([slim.batch_norm], is_training=is_regressor_training):
        if parallelize_heads:
            detect_logits = None
        regression_logits = _construct_pyramid(c5,
                                               detect_end_points,
                                               detect_logits,
                                               num_classes,
                                               original_scope_name,
                                               shared_net_scope_name + '_pyramid')

    return regression_logits, detect_end_points


def resnet_50_cascade(inputs,
                      num_classes=16,
                      is_detector_training=True,
                      is_regressor_training=True,
                      scope='resnet_cascade'):
    """TODO"""
    detect_logits, detect_endpoints = resnet_50_detector(inputs,
                                                         num_classes=num_classes,
                                                         is_training=is_detector_training,
                                                         scope='resnet_v1_50')
    detect_endpoints['detect_logits'] = detect_logits

    stacked_heatmaps = tf.concat(values=[detect_logits, inputs], axis=3)

    regression_logits, _ = resnet_50_detector(inputs=stacked_heatmaps,
                                              num_classes=num_classes,
                                              is_training=is_regressor_training,
                                              scope='resnet_50_regressor')

    return regression_logits, detect_endpoints


def resnet_152_detector(inputs,
                        num_classes=16,
                        is_detector_training=True,
                        is_regressor_training=True,
                        reuse=None,
                        scope='resnet_v1_152'):
    """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""

    blocks = [
        resnet_utils.Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block(
            'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        resnet_utils.Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        resnet_utils.Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]

    return bulat_resnet_v1(inputs,
                           blocks,
                           num_classes,
                           is_training=is_detector_training,
                           scope='resnet_v1_152')


def resnet_regressor(inputs, num_classes=16, is_training=True, scope='resnet_regressor'):
  """Hourglass model framed within residual learning from Bulat et al
     Descr: Compiles the bottleneck blocks and calls the hourglass function
  """
  # Add blocks pertain to the skiplayer blocks
  blocks = [
      resnet_utils.Block('block0', bottleneck, [(128, 64, 1)]*2 + [(256, 128, 1)]),
      resnet_utils.Block('block1', bottleneck, [(256, 128, 1)]*2 + [(512, 256, 1)]),
      resnet_utils.Block('block2', bottleneck, [(256, 128, 1)]*3),
      resnet_utils.Block('block3', bottleneck, [(256, 128, 1)]*2 + [(512, 256, 1)]),
      resnet_utils.Block('block4', bottleneck, [(256, 128, 1)]*3),
      resnet_utils.Block('block5', bottleneck, [(256, 128, 1)]*2 + [(512, 256, 1)]),
      resnet_utils.Block('block6', bottleneck, [(256, 128, 1)]*3),
      resnet_utils.Block('block7', bottleneck, [(256, 128, 1)]*2 + [(512, 256, 1)])
      ]
  return hourglass_bulat(inputs, blocks, scope=scope)


def hourglass_bulat(inputs, blocks, num_classes = 16, scope=None):

  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope(resnet_utils.resnet_arg_scope(), outputs_collection=end_points_collection):
      D0 = resnet_utils.conv2d_same(inputs, 64, 7, stride=2, scope='conv0')
      D0 = bottleneck(net, 128, 64, 3)
      D0 = slim.max_pool2d(net, [2, 2], stride=2, scope='pool0')
      D0 = resnet_utils.stack_blocks_dense(D0, [blocks[0]])
      skip0 = resnet_utils.stack_blocks_dense(D0, [blocks[1]])

      D1 = slim.max_pool2d(D0, [2, 2], stride=2, scope='pool1')
      D1 = resnet_utils.stack_blocks_dense(D1, [blocks[2]])
      skip1 = resnet_utils.stack_blocks_dense(D1, [blocks[3]])

      D2 = slim.max_pool2d(D1, [2, 2], stride=2, scope='pool2')
      D2 = resnet_utils.stack_blocks_dense(D2, [blocks[4]])
      skip2 = resnet_utils.stack_blocks_dense(D2, [blocks[5]])

      D3 = slim.max_pool2d(D2, [2, 2], stride=2, scope='pool3')
      D3 = resnet_utils.stack_blocks_dense(D3, [blocks[6]])
      skip3 = resnet_utils.stack_blocks_dense(D3, [blocks[7]])

      D4 = slim.max_pool2d(D3, [2, 2], stride=2, scope='pool4')
      D4 = bottleneck(D4, 512, 256, 3)
      D4 = slim.conv2d_transpose(D4, 512, [2,2], stride=2, scope='D4')
      D4 = D4 + skip3

      D5 = bottleneck(D4, 512, 256, 3)
      D5 = slim.conv2d_transpose(D5, 512, [2,2], stride=2, scope='D5')
      D5 = D5 + skip2

      D6 = bottleneck(D5, 512, 256, 3)
      D6 = slim.conv2d_transpose(D6, 512, [2,2], stride=2, scope='D6')
      D6 = D6 + skip1

      D7 = bottleneck(D6, 512, 256, 3)
      D7 = slim.conv2d_transpose(D7, 512, [2,2], stride=2, scope='D7')
      D7 = D7 + skip0

      D8 = slim.conv2d(D7, 512, [1,1], stride=1, scope='D8')
      D9 = slim.conv2d(D8, 256, [1,1], stride=1, scope='D9')
      D10 = slim.conv2d(D9, num_classes, [1,1], stride=1, scope='logits')
      D11 = slim.conv2d_transpose(D10, num_classes, [4, 4], stride=4, scope='logits_deconv')
      # Convert end_points_collection into a dictionary of end_points.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return D11, end_points
