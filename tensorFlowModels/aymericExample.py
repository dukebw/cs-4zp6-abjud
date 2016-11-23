from __future__ import print_function

import tensorflow as tf


# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 20
display_step = 10
num_parts = 1
num_channels = 3

# Network Parameters
img_size = 256
heat_map_size = 64
n_input = img_size*img_size # MNIST data input (img shape: 28*28)
n_output_size = heat_map_size * heat_map_size
#n_classes = 10 # MNIST total classes (0-9 digits)
#dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, num_channels])
x = tf.reshape(x, shape=[-1, img_size, img_size, num_channels])
print(x.get_shape())
y = tf.placeholder(tf.float32, [None, n_output_size])
y_heat_map = tf.reshape(y, [-1, heat_map_size, heat_map_size, num_parts])

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



# Construct model

print(weights['wc1'])
pred = conv_net(x, weights, biases)

print(pred.get_shape())
'''
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
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
'''
