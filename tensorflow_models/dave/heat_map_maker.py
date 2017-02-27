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


# constants

# file Information
database_access_file = "/home/david/programming/mcmaster-text-to-motion-database/tensorflow_models/dave/pose_DB.sqlite"
data_path = "/home/david/FLIC-full/images/"



# Parameters
batch_length = 5


# heat map parameters
num_parts = 7
heat_map_size = 5


# Gaussian Constants
gauss_deviation = 0.5
norm_factor = 2*math.pi*gauss_deviation*gauss_deviation
norm_factor = 1/norm_factor


# Images Constants
num_channels = 3
img_size = 256


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases):

    # specifics of the model from caffe
    # blobs_lr: 1; blobs_lr: 2; -- individualize the learning rates weights
    # vs biases
    # weight_decay: 1; weight_decay: 0; these are the regularization terms
    # not sure how to specify this in tensorFlow?   
    # num_output = 128; kernel_size = 5; stride : 1; pad: 2; basic parameters
    # weight_filler {type: gaussian std: 0.01 -- setting the initial weights
    # bias filler: type constant; value 0
    # then there's a rectilinear unit and max_pooling of size 2

    # not sure about this but since there are 3 channels in the jpg
    # this means the input dimension should be 3
    # Reshape input picture
    #x = tf.reshape(x, shape=[-1, img_size, img_size, num_channels, 1])

    # Convolution Layer
    print(x.get_shape())
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    conv1 = tf.nn.relu(conv1)
    print(conv1.get_shape())
    # test this!

    # Convolution Layer2
    
    # seems like the same as layer 1 except the number of inputs is different
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    conv2 = tf.nn.relu(conv2)
    print(conv2.get_shape())


    # layer3
    # takes input from pool2
    # blobs_lr 1 blobs_lr 2
    # weight decay 1 ; weight decay 0
    # num_output 128
    # kernel_size 5
    # relu
    # initial weight filler is gaussian deviation is 0.01
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = tf.nn.relu(conv3)
    print(conv3.get_shape())

 


    # layer4
    # num output 256
    # kernel size is 9
    # pad is 4
    # bias filler is 1 (constant)
    # blobs_lr 1 blobs_lr 2
    # weight decay 1 ; weight decay 0
    # num_output 128
    # kernel_size 5
    # relu
    # initial weight filler is gaussian deviation is 0.01
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = tf.nn.relu(conv4)
    print(conv4.get_shape())


    # layer5
    # num output 512
    # kernel size is 9
    # pad is 4
    # bias filler is 1 (constant)
    # blobs_lr 1 blobs_lr 2
    # weight decay 1 ; weight decay 0
    # num_output 128
    # kernel_size 5
    # relu
    # initial weight filler is gaussian deviation is 0.01
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    conv5 = tf.nn.relu(conv5)
    print(conv5.get_shape())



    # layer6
    # num output 256
    # kernel size is 1
    # pad is 4
    # bias filler is 1 (constant)
    # blobs_lr 1 blobs_lr 2
    # weight decay 1 ; weight decay 0
    # num_output 128
    # kernel_size 5
    # relu
    # initial weight filler is gaussian deviation is 0.01
    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'])
    conv6 = tf.nn.relu(conv6)


    # layer7
    # num output 512
    # kernel size is 9
    # pad is 4
    # bias filler is 1 (constant)
    # blobs_lr 1 blobs_lr 2
    # weight decay 1 ; weight decay 0
    # num_output 128
    # kernel_size 5
    # relu
    # initial weight filler is gaussian deviation is 0.01
    conv7 = conv2d(conv6, weights['wc7'], biases['bc7'])
    conv7 = tf.nn.relu(conv7)


    
    # layer8
    # num output 7
    # kernel size is 1
    # pad is 4
    # bias filler is 1 (constant)
    # blobs_lr 1 blobs_lr 2
    # weight decay 1 ; weight decay 0
    # num_output 128
    # relu
    # initial weight filler is gaussian deviation is 0.01
    conv8 = conv2d(conv7, weights['wc8'], biases['bc8'])
    conv8 = tf.nn.relu(conv8)

    # I take this transpose so that the output has the same shape as the heatmap.
    conv8 = tf.transpose(conv8, perm=[0,3,1,2])
    print(conv8.get_shape())
    return conv8

    
  


# Store layers weight & bias
weights = {
    # layer1 5x5 conv, 1 input, 128 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 128])),
    # layer2 5x5 conv, 128 inputs, 128 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 128, 128])),
    # layer3 5x5 conv, 128 inputs, 128 outputs
    'wc3': tf.Variable(tf.random_normal([5, 5, 128, 128])),
    # layer4 5x5 conv, 128 inputs, 128 outputs
    'wc4': tf.Variable(tf.random_normal([5, 5, 128, 256])),
    # layer5 5x5 conv, 128 inputs, 128 outputs
    'wc5': tf.Variable(tf.random_normal([9, 9, 256, 512])),
    # layer6 5x5 conv, 128 inputs, 128 outputs
    'wc6': tf.Variable(tf.random_normal([1, 1, 512, 256])),
    # layer6 5x5 conv, 128 inputs, 128 outputs
    'wc7': tf.Variable(tf.random_normal([1, 1, 256, 256])),
    # layer7 5x5 conv, 128 inputs, 128 outputs
    # not sure if I should flatten this and make the last layer
    # fully connected?
    'wc8': tf.Variable(tf.random_normal([1, 1, 256, 7]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([128])), # set to constant value 0
    'bc2': tf.Variable(tf.random_normal([128])), # 1 should be set to constant of 1
    'bc3': tf.Variable(tf.random_normal([128])), # 1
    'bc4': tf.Variable(tf.random_normal([256])), # 1
    'bc5': tf.Variable(tf.random_normal([512])), # 1
    'bc6': tf.Variable(tf.random_normal([256])), # 1
    'bc7': tf.Variable(tf.random_normal([256])), 
    'bc8': tf.Variable(tf.random_normal([7]))
}


# here I read from the sqlite file that is in the parent folder
# the purpose is to get a list of filenames for the batch
# and an np-array of x and y coordinates.
# it's important that they have the specific shape below
# because they are used in matrix multiplication later on.
def read_label_file(file, path, num_parts, image_dimension):
    numberOfFiles = 20
    #f = open(file, 'r')
    print(file)
    conn = sqlite3.connect(file)
    c = conn.cursor()
    # note that can't adjust the number of files.Have to return multipes of 20 -- you can see why if
    # you look at the line below where the 20 is encoded into the SELECT statement.
    c.execute("SELECT filename, right_x_shoulder, right_y_shoulder, left_x_shoulder, left_y_shoulder, right_x_wrist, right_y_wrist, left_x_wrist, left_y_wrist, left_x_elbow, left_y_elbow, right_x_elbow, right_y_elbow, x_head, y_head FROM pose_train_table WHERE file_key \
            < 20")#, (numberOfFiles))
    conn.commit()
    #row = c.fetchone()    
    filepaths = []
    x = np.empty([numberOfFiles, num_parts,1])
    y = np.empty([numberOfFiles, num_parts,1])
    #labels = [] # these labels will be 14 numbers but for now you could just use the shoulder data
    fetch = c.fetchmany
    rows = fetch(numberOfFiles)
    count = 0;
    for row in rows:
        filepaths.append(path + row[0].strip())
        # here I get factors that are used to rescale the x and y body positions because
        # the jpeg will be reshaped to image_dimesion * image_dimension
        width, height = get_jpeg_dimensions(path + row[0].strip())
        #rescale_the_image(path + row[0].strip(),path + 'rescale_'+row[0].strip())
        width_factor = image_dimension/width
        length_factor = image_dimension/height
        for part_num in range(0, num_parts):
            print("this is a joint number" )
            print(part_num)
            x[count, part_num, 0] = width_factor * row[2*part_num+1]
            y[count, part_num, 0] = length_factor * row[2*part_num+2]
            print(x[count, part_num, 0])
            print(y[count, part_num, 0])
        count = count +1
    return filepaths, x, y

# function gets the scaling factors for each file used in the batch
# to scale the joint positinos
def get_jpeg_dimensions(path_to_file):
    im = Image.open(path_to_file)
    return im.size

# this is a constant of the form (1,2,3,4... length)
# it is used because in the generation of the heat maps we need terms
# x-x_avg where x is the pixel location so this vector is used to keep track of the pixel value
def create_index_constant(length):
    z = np.empty([1,length])
    for x in range(1,length+1):
        z[0,x-1] = x
    return z



def _input_pipeline(filename, data_path, batch_size, image_dimension, rescale_size, num_channels, num_parts, stdev, gaussFact, num_epochs=None):

    # gets a list of files and x and y np arrays
    image_list, label_list_x, label_list_y = read_label_file(filename, data_path, num_parts, image_dimension)

    # converts the above into tensors
    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels_x = ops.convert_to_tensor(label_list_x, dtype=dtypes.int32)
    labels_y = ops.convert_to_tensor(label_list_y, dtype=dtypes.int32)
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels_x, labels_y],
                                                num_epochs=num_epochs,
                                                shuffle=True)

    # deques the queue and decodes the filenames into data which is returned as a tensor
    # also in this function the images are rescaled to be image_dimension squared
    image, label_x, label_y = read_images_from_disk(input_queue, rescale_size, num_channels)
    pr_image = image #processing_image(image)
    # transforms the image np array of x coordinates into
    # an array of shape (image_dimension, image_dimension, num_parts)
    # for each part the i, j th entry is
    # i - x_(avg for that part) -- note this is independent of j
    pr_label_x = processing_labels(label_x, num_parts, image_dimension, 1)
    # this is the same as above but for the y positions so value is independent of i not j
    pr_label_y = processing_labels(label_y, num_parts, image_dimension, 0)
    # these two tensors are added together and exponentiated into a gaussian
    pr_label = combinexy(pr_label_x, pr_label_y, stdev, gaussFact)
    
    # batches are created
    image_batch, label_batch = tf.train.batch([pr_image, pr_label],
                                              batch_size=batch_size)
    
    # Display the training images in the visualizer.
    tensor_name = image.op.name
    tf.image_summary(tensor_name + 'images', image_batch)
    return image_batch, label_batch


# takes a np array of size (num_parts,1)
# outputs on of size [num_parts,image_dimension, image_dimension]
# each entry for each part should be (x - x_avg) ^2 for all body parts and
# all x values
def processing_labels(labelx, num_parts, image_dimension, xory):

    # create a column vector of ones
    b = tf.ones([1,image_dimension], tf.int32)
    # multiply this column of ones by the input to get an output of a
    # matrix where each column is of size image_dimension.  The matrix has the same value
    # over the entire column (the x value for that joint)
    outtx = tf.matmul(labelx,b)
    # step 2 is to create a matrix that is of size
    # num_parts * image_dimension but whose value denotes the x-coordinate
    # it's again costant over the columns
    c1  = create_index_constant(image_dimension)
    c = tf.constant(c1, dtype=np.int32)
    d = tf.ones([num_parts,1], tf.int32)
    other = tf.matmul(d, c)

    # now we are getting the value (x - x_avg) ^2
    squared_x = tf.squared_difference(outtx, other,name=None)
    # this is where things get tricky and I just went by trial and error until
    # a tensor was printed on the screen that was arranged properly
    squared_x = tf.reshape(squared_x, [1,num_parts*image_dimension])
    e = tf.ones([image_dimension, 1], tf.int32)
    final_x = tf.matmul(e, squared_x)
    final_x = tf.reshape(final_x, [image_dimension, image_dimension*num_parts])
    final_x = tf.transpose(final_x)
    final_x = tf.reshape(final_x, [num_parts,image_dimension, image_dimension])
    # if we have the x positions passed we have to perform a transpose so that they
    # are constant in the vertical direction.
    # for the y -coordinate we don't have to do this.
    if xory == 1:
        final_x = tf.transpose(final_x, perm=[0,2,1])
    final_x = tf.cast(final_x, tf.float32)
    return final_x


# have as input (x-x_avg)^2 and (y-y_avg)^2
# so combine these into gaussians e ^-((x-x_avg)^2 and (y-y_avg)^2)
def combinexy(labelx, labely, stdev, gaussFact):
    outt = tf.add(labelx, labely, name=None)
    outt = tf.scalar_mul(-1/stdev, outt)
    outt = tf.exp(outt, name=None)
    outt = tf.scalar_mul(gaussFact, outt)
    return outt


def read_images_from_disk(input_queue, rescale_size, num_channels):
    """Consumes a single filename and label as a ' '-delimited string.
    Parameters
    ----------
      filename_and_label_tensor: A scalar string tensor.
    Returns
    -------
      Two tensors: the decoded image, and the string label.
    """
    label_x = input_queue[1]
    label_y = input_queue[2]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=num_channels)
    example = rescale_image(example, rescale_size, num_channels)
    # processed_label = label
    return example, label_x, label_y

def rescale_image(image, rescale_size, num_channels):
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
    dimensions = rescale_size*dimensions
    
    resized_image = tf.image.resize_images(image, dimensions, method=0)
    
    resized_image.set_shape([rescale_size, rescale_size, num_channels])
    return resized_image

x = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])
x = tf.reshape(x, [-1, img_size, img_size, num_channels])
print(x.get_shape())
y = tf.placeholder(tf.float32, [None, num_parts, heat_map_size, heat_map_size])
y = tf.reshape(y, [-1, num_parts, heat_map_size, heat_map_size])
#sets_image = tf.placeholder(tf.float32, [None, n_input, img_size, img_size, num_channels])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

pred = conv_net(x, weights, biases)
#loss = tf.reduce_mean(cross_entropy, name=None)





def test_pipeline(database_access_file, data_path,  batch_length, image_dimension, rescale_size, num_channels, num_parts, stdev, gaussFact):
    #_input_pipeline(filename, data_path, batch_size, num_parts, stdev, gaussFact, num_epochs=None)
    # image_batch, label_batch = inputs(filename, 75, 2, data_folder)
    image_batch, label_batch = _input_pipeline(database_access_file, data_path,  batch_length, image_dimension, rescale_size, num_channels, num_parts, stdev, gaussFact)

    pred = conv_net(image_batch, weights, biases)
    #outt = tf.squared_difference(pred, label_batch,name=None)
    # minThis = tf.nn.l2_loss(outt, name=None)
    # the image batch would be a set of tensors that are of size
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    with sess.as_default():
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #result = sess.run(loss ,feed_dict={x: image_batch, y:label_batch })
        print(sess.run(label_batch))
        print(sess.run(pred))
        #print(sess.run(minThis))
        
        #image_batch, label_batch = _input_pipeline(database_access_file, data_path,  batch_length, image_dimension, rescale_size, num_channels, num_parts, stdev, gaussFact)

        #image_batch, label_batch = _input_pipeline(database_access_file, data_path,  batch_length, image_dimension, rescale_size, num_channels, num_parts, stdev, gaussFact)
       
        #result = sess.run(output ,feed_dict={y: get_heat_map_data(limb_points_o), x:getImageData() })
        #print(sess.run(tf.diag(create_index_constant(2))))
        #result = sess.run(outty ,feed_dict={ a:create_index_constant(64)})
        #print(outty.get_shape())
        coord.request_stop()
        coord.join(threads)
        print("Finish Test")
        sess.close()



# the first thing to do is to put some of this data through a network.
# that would be sweet.

if __name__ == '__main__':
    test_pipeline(database_access_file, data_path,  batch_length, heat_map_size, img_size, num_channels, num_parts, gauss_deviation, norm_factor)
