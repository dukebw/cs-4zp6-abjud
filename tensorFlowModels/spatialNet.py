'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters


#things have to learn from the paper.  *****************
# the specific numbers they use



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
