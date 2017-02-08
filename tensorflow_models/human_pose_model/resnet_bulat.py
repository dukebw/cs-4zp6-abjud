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

import tensorflow as tf

import resnet_utils

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope

'''

'''
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

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
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')

    output = shortcut + residual

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)


def bulat_resnet_v1(inputs,
                    blocks,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    include_root_block=True,
                    reuse=None,
                    scope=None):
  """
  "Blocks B1-B4 are the same as the ones in the original ResNet, and B5 was slightly modified. We firstly removed both
  the fully connected layer after B5 and then the preceding average pooling layer. Then, we added a scoring
  convolutional layer B6 with N outputs, one for each part. Next, to address the extremely low output resolution, we
  firstly modified B5 by changing the stride of its convolutional layers from 2px to 1px and then added (after B6) a
  deconvolution layer B7 with a kernel size and stride of 4, that upsamples the output layers to match the resolution
  of the input."
  """
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope(resnet_utils.resnet_arg_scope(),[slim.conv2d, bottleneck, resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        # We do not include batch normalization or activation functions in
        # conv1 because the first ResNet unit will perform these. Cf.
        # Appendix of [2].
        # B1
        # 1x Conv       64, 7x7, 2x2
        # 1x Pooling    3x3, 2x2
        net = resnet_utils.conv2d_same(net, 64, [7,7], stride=2, scope='conv1', activation_fn=None, normalizer_fn=None)
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        # B2 - B5
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        net = slim.conv2d(net, num_classes, [1, 1], scope='logits')
        net = slim.conv2d_transpose(net, num_classes, [4, 4], stride=4, scope='b7_deconv')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points


def resnet_detector(inputs,
                    num_classes=16,
                    is_training=True,
                    reuse=None,
                    scope='resnet_v1_152'):
  """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""

  blocks = [
      # B2
      resnet_utils.Block('block1', bottleneck, [(256, 64, 1)] * 3),
      # B3
      resnet_utils.Block('block2', bottleneck, [(512, 128, 1)] * 8),
      # B4
      resnet_utils.Block('block3', bottleneck, [(1024, 256, 1)] * 36),
      # B4
      # ADDED TO EXCLUDE PROPERLY with namescope as this is a modification from the original resnet
      resnet_utils.Block('block3b', bottleneck, [(1024, 256, 1)] * 2),
      # B5
      resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
  return bulat_resnet_v1(inputs, blocks, num_classes, is_training=is_training,
                         include_root_block=True, reuse=reuse, scope=scope)


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

