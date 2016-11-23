'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


# you will get as input a 28x28 sized image.
# a list of 7 key joint positions

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters


#things have to learn from the paper.  *****************
# the specific numbers they use


# to get the data step 1 is to put the data into

# put the data into a tensor of size (num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
# then you have a list of labels -- something like ideal output

# steps seem to be: 


# set the names of the datapaths

dataset_path = ""
test_labels_file = ""
train_labels_file = ""


# probably want to change most of this
test_set_size = 5
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CHANNELS = 3


# Creating a pipeline step 1
# Load the label data which contains lists
# of input files

def read_label_file(file):
    f = open(file, 'r')
    filepaths = []
    labels = []
    for line in f:
        filepath, label = line.split(",") # this is for csv files
        filepaths.append(filepath)
        labels.append(encode_label(label))
    return filepaths, labels

def encode_label(label):
    return int(label)

# here call both of the above methods
train_filepaths, train_labels = read_label_file(dataset_path + train_labels_file)
test_filepaths, test_labels = read_label_file(dataset_path + test_labels_file)

# those are just filenames so next step is to turn them into filepaths
# combine test and train into a list of all


# Step 2 is to do some optional processing on our string lists
# currently choosing not to bother with this option


# Step 3 start building the pipeline
# they say the data type of the tensor must match that of the lists
# not sure what this means at the moment

all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)


#make partitions

partitions = [0]*len(all_filepaths)
partitions[:test_set_size]=[1]*test_set_size
random.shuffle(partitions)

# partition the data into a test and train set according to our
# partition vector
train_images, test_images = tf.dynamic_partition(all_images, partitions,2)
train_labels, test_labels = tf.dynamic_partition(all_labels, partitions,2)


# Create input Queues and defint how to load the images

train_input_queue = tf.train.slice_input_producer(
    [train_images, train_labels],
    shuffle=False)
test_input_queue = tf.train.slice_input_producer(
    [test_images,test_labels],
    shuffle=False)

# process and string tensor into an image and a label

file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
test_label = test_input_queue[1]

# Group Samples into Batches
# if run train_image in a session you would get a single image (28,28,1)
# training 1 at a time is inefficient so want to work in batches

# define tensor shape

train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

#collect batches of images before processing
train_image_batch, train_label_batch = tf.train.batch(
    [train_image,train_label],
    batch_size=BATCH_SIZE
    #,num_threads =1
    )

test_image_batch, test_label_batch = tf.train.batch(
    [test_image,test_label],
    batch_size=BATCH_SIZE
    #,num_threads =1
    )

# Run the queue runners and start a session
# finished building the input pipeline however if try to acccess
# 

# for each picture used in training I need 7 coordinates -- one for each joint.
# create a family of 7 Gaussians using these 7 points

learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # needs to change not sure the size of the images
n_classes = 7 # There are 7 body parts currently that we are considering
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


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

# all the specifics of the model from caffe

# blobs_lr: 1; blobs_lr: 2; -- individualize the learning rates weights
# vs biases
# weight_decay: 1; weight_decay: 0; these are the regularization terms
# not sure how to specify this in tensorFlow?
# num_output = 128; kernel_size = 5; stride : 1; pad: 2; basic parameters
# weight_filler {type: gaussian std: 0.01 -- setting the initial weights
# bias filler: type constant; value 0
# then there's a rectilinear unit and max_pooling of size 2

# implement this



# layer2
# seems like the same as layer 1 except the number of inputs is different

# layer3
# details
# seems the same as above again but there's no pooling

# layer4
# 
#


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convulution Layer 3
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    
    # Convulution Layer 4
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    
    # Convulution Layer 5
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    
    # Convulution Layer 6
    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'])
    
    # Convulution Layer 7
    conv7 = conv2d(conv6, weights['wc7'], biases['bc7'])

    # Convulution Layer 8
    conv7 = conv2d(conv7, weights['wc8'], biases['bc8'])
    
    # Apply Dropout
    conv8 = tf.nn.dropout(conv8, dropout)

    return conv8

# Store layers weight & bias
weights = {


    # note I'm sure that these input numbers are wrong
    # think they mean the number of channels that are connected together
    
    
    # layer1 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 128])),
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
    'wc8': tf.Variable(tf.random_normal([1, 1, 256, 7])),

}

biases = {
    'bc1': tf.Variable(tf.random_normal([128])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([128]),                        
    'bc4': tf.Variable(tf.random_normal([256]),
    'bc5': tf.Variable(tf.random_normal([512]),
    'bc6': tf.Variable(tf.random_normal([256]),
    'bc7': tf.Variable(tf.random_normal([256]),
                        
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# it's hard to define the loss

# the output will be a 1 x 1 x 7 Google tensor which need to use to calculate
# the coordinates are 1 x 1 part and x 7 part is the body part
# Gauss ((x1 - i) ^2 + (x2 - y)^2 ) - psi(x1,x2) )^2 -- where we sum over all
# body parts.  To calculate we probably need to flatten or reshape



# Define loss and optimizer
# need to calculate Gijk(yk)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
cost = tf.reduce_mean(tf.square(     ))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1


    # this part needs to be completely re-written
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict = fill_feed_dict()
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
keep_prob: 1.}))
