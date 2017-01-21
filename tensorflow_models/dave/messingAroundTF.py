# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:50:47 2015
@author: teichman
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import numpy as np


import gzip
import os
import re
import sys
import zipfile
import random
import math
import logging
import scipy as scp
import scipy.misc
from six.moves import urllib

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.training import queue_runner

from tensorflow.python.ops import random_ops

# Global constents descriping data set

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)







def _input_pipeline(filename, batch_size, num_labels, num_epochs=None):
                    #processing_image=lambda x: x,
                     #processing_label=lambda y: y,
                    #num_epochs=None):
    """The input pipeline for reading images classification data.
    The data should be stored in a single text file of using the format:
     /path/to/image_0 label_0
     /path/to/image_1 label_1
     /path/to/image_2 label_2
     ...
     Args:
       filename: the path to the txt file
       batch_size: size of batches produced
       num_epochs: optionally limited the amount of epochs
    Returns:
       List with all filenames in file image_list_file
    """

    # Reads pfathes of images together with there labels
    image_list, label_list = read_labeled_image_list(filename)

    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True)

    # Reads the actual images from
    image, label = read_images_from_disk(input_queue, num_labels=num_labels)
    pr_image = image #processing_image(image)
    pr_label = label #processing_label(label)

    image_batch, label_batch = tf.train.batch([pr_image, pr_label],
                                              batch_size=batch_size)
    print(image_batch)
    print(label_batch)

    # Display the training images in the visualizer.
    tensor_name = image.op.name
    tf.image_summary(tensor_name + 'images', image_batch)
    return image_batch, label_batch




def read_images_from_disk(input_queue, num_labels):
    """Consumes a single filename and label as a ' '-delimited string.
    Parameters
    ----------
      filename_and_label_tensor: A scalar string tensor.
    Returns
    -------
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    print("this is the shape of the queue")
    print(label.get_shape())
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    example = rescale_image(example)
    # processed_label = label
    return example, label


def rescale_image(image):
    """
    Resizes the images.
    Parameters
    ----------
    image: An image tensor.
    Returns
    -------
    An image tensor with size H['arch']['image_size']
    """
    dimensions = tf.ones([2], tf.int32)
    dimensions = 256*dimensions
    print(dimensions)

    resized_image = tf.image.resize_images(image, dimensions, method=0) #H['arch']['image_size'],
                                           #H['arch']['image_size'], method=0)
    resized_image.set_shape([256, 256, 3])
    return resized_image


def create_one_hot(label, num_labels=10):
    """
    Produces one_hot vectors out of numerical labels
    Parameters
    ----------
       label_batch: a batch of labels
       num_labels: maximal number of labels
    Returns
    -------
       Label Coded as one-hot vector
    """

    labels = tf.sparse_to_dense(label, [num_labels], 1.0, 0.0)

    return labels


def read_labeled_image_list(image_list_file):
    """
    Read a .txt file containing pathes and labeles.
    Parameters
    ----------
     image_list_file : a .txt file with one /path/to/image per line
     label : optionally, if set label will be pasted after each line
    Returns
    -------
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels



def test_preprocc():
    data_folder = "/home/david"
    data_file = "labelContainer"

    filename = os.path.join(data_folder, data_file)

    image_list, label_list = read_labeled_image_list(filename)

    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=None,
                                                shuffle=True)

    image, label = read_images_from_disk(input_queue, num_labels=2)

    reshaped_image = random_resize(image, 32,
                                   48)

    init_op = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reshaped_image.eval(session=sess).shape

    new_size = tf.to_int32(random_ops.random_uniform([], 32, 45))
    rimage = tf.image.resize_images(image, new_size, new_size,
                                    method=0)

    a = sess.run([new_size, rimage])
    print(a[0])
    a[1].shape
    return

# going to have a placeholder with a shape that's
# lets say (N, num_parts, x, y)
# constant tensor of shape 256 x 256 which
# nice if you could create a constant tensor of type (i,j) = i
#                                   second one where (i,j) = j
# new = x * constant Tensor

# def createGaussMap():
    

def test_pipeline():
    data_folder = "/home/david"
    data_file = "labelContainer"

    filename = os.path.join(data_folder, data_file)


    print(filename)
    #image_batch, label_batch = inputs(filename, 75, 2, data_folder)
    batch_size = 5
    num_labels = 1
    label_batch = _input_pipeline(filename, batch_size, num_labels)
    # Create the graph, etc.
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    with sess.as_default():
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print(sess.run(label_batch))

        coord.request_stop()
        coord.join(threads)

        print("Finish Test")

        sess.close()


def maybe_download_and_extract(H, dest_directory):
    """Download and extract Data found in data_url."""
    return

# the first thing to do is to put some of this data through a network.
# that would be sweet.

if __name__ == '__main__':
    # test_one_hot()
    #test_preprocc()
    test_pipeline()
    
