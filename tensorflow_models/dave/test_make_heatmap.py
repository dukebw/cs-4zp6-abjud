from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import threading
import math
import sqlite3
import PIL
from PIL import Image
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from random import randint


import os.path
import time
import numpy as np


import gzip
import os
import re
import sys
import zipfile
import logging
import scipy as scp
import scipy.misc
from six.moves import urllib

import tensorflow as tf
from tensorflow.python.training import queue_runner

from tensorflow.python.ops import random_ops

# it would be nice to create a program that
# it a function and takes as input a placeholder
# and outputs a heatmap.

dataset_path = "/home/david/FLIC-full/images/"
# maybe I should have a separate database with the test labels?
train_labels_file = "pose_DB.sqlite"
#b = tf.ones([1,64], tf.float32)

#a = tf.placeholder(tf.float32, [64])
#a = tf.reshape(a,[1,64])
#outty = tf.matmul(a, b)



# create a variable that has the values 1, 2,3,4, 5, ...64 in it

def create_index_constant(length):
    z = np.empty([1,length])
    for x in range(1,length+1):
        z[0,x-1] = x
    return z

# the only way to put

# going to have a placeholder with a shape that's
# lets say (N, num_parts, x, y)
# constant tensor of shape 256 x 256 which
# nice if you could create a constant tensor of type (i,j) = i
#                                   second one where (i,j) = j
# new = x * constant Tensor


def _input_pipeline(filename, data_path, batch_size, num_labels, num_epochs=None):
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
    image_list, label_list_x, label_list_y = read_label_file(filename, dataset_path)

    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels_x = ops.convert_to_tensor(label_list_x, dtype=dtypes.int32)
    labels_y = ops.convert_to_tensor(label_list_y, dtype=dtypes.int32)
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels_x, labels_y],
                                                num_epochs=num_epochs,
                                                shuffle=True)

    # Reads the actual images from
    image, label_x, label_y = read_images_from_disk(input_queue, num_labels=num_labels)
    pr_image = image #processing_image(image)
    pr_label = processing_labels(label_x, label_y, 7)
    # this is interesting have an opportunity to alter the label here which we'll have
    # to do.

    image_batch, label_batch = tf.train.batch([pr_image, pr_label],
                                              batch_size=batch_size)

    print(image_batch)
    # Display the training images in the visualizer.
    tensor_name = image.op.name
    tf.image_summary(tensor_name + 'images', image_batch)
    return image_batch, label_batch


def processing_labels(labelx, num_parts, image_dimension, xory):
    # step 1 is create a tensor that is (1,64)
    #
    #a = create_index_constant(image_dimension)
    b = tf.ones([1,image_dimension], tf.int32)
    #b = tf.constant(a, dtype=np.int32)
    outtx = tf.matmul(labelx,b)
    # step 2 is create a (num_parts, 64) sized array
    # create a matrix that is num_parts by image_dimesion in size
    # 

    # c1 is like b above
    c1  = create_index_constant(image_dimension)
    c = tf.constant(c1, dtype=np.int32)
    d = tf.ones([num_parts,1], tf.int32)
    other = tf.matmul(d, c)
    
    squared_x = tf.squared_difference(outtx, other,name=None)
    
    squared_x = tf.reshape(squared_x, [1,num_parts*image_dimension])

    
    e = tf.ones([image_dimension, 1], tf.int32)
    final_x = tf.matmul(e, squared_x)
    final_x = tf.reshape(final_x, [image_dimension, image_dimension*num_parts])
    final_x = tf.transpose(final_x)
    final_x = tf.reshape(final_x, [num_parts,image_dimension, image_dimension])
    if xory == 1:
        final_x = tf.transpose(final_x, perm=[0,2,1])
    final_x = tf.cast(final_x, tf.float32)
    return final_x


def combinexy(labelx, labely, stdev, gaussFact):
    outt = tf.add(labelx, labely, name=None)
    outt = tf.scalar_mul(stdev, outt)
    outt = tf.exp(outt, name=None)
    outt = tf.scalar_mul(gaussFact, outt)
    return outt
    
    

# here I am just making a list of part positions which will be used to
# test the heat map
first_label = np.empty([3,1])
for j in range(0,3):
    first_label[j,0] = j + 3

 
'''
first_label[0,0] = 6
first_label[1,0] = 55
first_label[2,0] = 0  
'''
test_x = tf.constant(first_label, dtype=np.int32)

test_1 = processing_labels(test_x, 3, 4, 0)
test_2 = processing_labels(test_x, 3, 4, 1)
test_3 = combinexy(test_1, test_2, 0.05, 3)

def test_pipeline():

    
    # the image batch would be a set of tensors that are of size
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    with sess.as_default():
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        result = sess.run(test_3)
        print(result)
        #print(sess.run(tf.diag(create_index_constant(2))))
        #result = sess.run(outty ,feed_dict={ a:create_index_constant(64)})
        #print(outty.get_shape())
        coord.request_stop()
        coord.join(threads)
        print("Finish Test")
        sess.close()



# could create a constant for this.



# the first thing to do is to put some of this data through a network.
# that would be sweet.

if __name__ == '__main__':
    test_pipeline()
