from __future__ import print_function
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



dataset_path = "/home/david/FLIC-full/images/"

# maybe I should have a separate database with the test labels?

train_labels_file = "pose_DB.sqlite"

# Generating some simple data
r = np.arange(0.0,100003.0)
#print(r.shape)

# so there's set of r,r,r,r -> r is defined above
# and so you get 
raw_data = np.dstack((r,r,r,r))[0]
data_size_label = 25
data_size_input = 4
num_in_batch = 20

gauss_deviation = 10
norm_factor = 2*math.pi*gauss_deviation*gauss_deviation
norm_factor = 1/norm_factor

#print(raw_data.shape)
raw_target = np.array([[1,0,0]] * 100003)

# Parameters
learning_rate = 0.001
training_iters = 40
batch_size = 20
display_step = 10
num_parts = 7
num_channels = 3

# Network Parameters
img_size = 256
heat_map_size = 64
size_input = img_size*img_size*num_channels # MNIST data input (img shape: 28*28)
n_output_size = heat_map_size * heat_map_size
#n_classes = 10 # MNIST total classes (0-9 digits)
#dropout = 0.75 # Dropout, probability to keep units


# are used to feed data into our queue

queue_input_data = tf.placeholder(tf.float32, shape=[20, size_input])
queue_input_target = tf.placeholder(tf.float32, shape=[20, n_output_size])
queue = tf.FIFOQueue(capacity=200, dtypes=[tf.float32, tf.float32], shapes=[[size_input], [n_output_size]])
enqueue_op = queue.enqueue_many([queue_input_data, queue_input_target])
dequeue_op = queue.dequeue()

# tensorflow recommendation:
# capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=1, capacity=40)
# use this to shuffle batches:
# data_batch, target_batch = tf.train.shuffle_batch(dequeue_op, batch_size=15, capacity=40, min_after_dequeue=5)


def encode_gauss_map(x_avg, y_avg): # here I should create a 64x64 sized array
    labels = []
    for x in range(0, heat_map_size):
        for y in range(0, heat_map_size):
            value = 2*gauss_deviation**2
            value = ((x - x_avg)**2 + (y - y_avg)**2)/value
            value = norm_factor * math.e **(-value)
            labels.append(value)
    return np.array(labels).reshape(64,64)

def enqueue():
  """ Iterates over our data puts small junks into our queue."""
  under = 0
  max = 39 #len(raw_data)
  print(max)
  print(under+20)
  image_data_list = []
  while under < (max -20):
    print("starting to write into queue")
    upper = under + 20
    print("try to enqueue ", under, " to ", upper)
    if upper <= max:
        curr_target = np.empty([num_parts, heat_map_size, heat_map_size])
        #for x in range(0,20):
        # first step is get a list of 20 filepaths and 20* 7 coordinates
        # that can use to put in the Gauss map
        # step 1 is to get the filepaths

        filepaths, limb_points = read_label_file(train_labels_file, dataset_path)
        #filename_queue = tf.train.string_input_producer(filepaths)

        # step 2 is to load these into a
        # step 3 is to put the data from each of these into
        # now have to loop over each file in the queue and put that
        #
        for x in range(0, len(filepaths)):
            print("started the queue")
            reader = tf.WholeFileReader()
            key, value = reader.read(filename_queue)
            print("read the queue")
            my_img = tf.image.decode_jpeg(value)
            print("decoded the queue")
            #t = my_img.eval()
            #t.reshape([1, 256, 256, 3])
            print("evaluation complete")
            for y in range(0,num_parts):
                curr_target[y, 0:64, 0:64] = encode_gauss_map(limb_points[2*y],limb_points[2*y+1])
            sess.run(enqueue_op, feed_dict={queue_input_data: my_img.eval().reshape([1, 256, 256, 3]),
                                    queue_input_target: curr_target})
            
    print("added to the queue")
    #if sess.should_stop():
    #    print("Trying to BBBBBREAK")
    #    break
  print("finished enqueueing")

def retrieveData(file):
    file_content = tf.read_file(file)
    train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    return train_image

def rescale_the_image(filePath, newName):
    basewidth = 256
    img = Image.open(filePath) #'/home/david/FLIC-full/images/2-fast-2-furious-00003571.jpg')
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth, basewidth), PIL.Image.ANTIALIAS)
    img.save(newName) #'resized_image.jpg')

# need to get all 7 body parts somehow put into the labels
#

def read_label_file(file, path):
    numberOfFiles = 20
    #f = open(file, 'r')
    conn = sqlite3.connect(file)
    c = conn.cursor()
    # note that can't adjust the number of files.Have to return multipes of 20 -- you can see why if
    # you look at the line below where the 20 is encoded into the SELECT statement.
    c.execute("SELECT filename, right_x_shoulder, right_y_shoulder, left_x_shoulder, left_y_shoulder, right_x_wrist, right_y_wrist, left_x_wrist, left_y_wrist, left_x_elbow, left_y_elbow, right_x_elbow, right_y_elbow, x_head, y_head FROM pose_train_table WHERE file_key \
            < 20")#, (numberOfFiles))
    conn.commit()
    #row = c.fetchone()    
    filepaths = []
    part_list = []
    #labels = [] # these labels will be 14 numbers but for now you could just use the shoulder data
    fetch = c.fetchmany
    rows = fetch(numberOfFiles)
    for row in rows:
        print ("it's going")
        filepaths.append(dataset_path + 'rescale_' + row[0].strip())
        rescale_the_image(path + row[0].strip(),path + 'rescale_'+row[0].strip())
        for part_num in range(0, num_parts):
            #print(part_num)
            part_list.append(row[2*part_num+1])
            part_list.append(row[2*part_num+2])
    return filepaths, part_list

# the imanolsPipeline ends here
# tf Graph input
x = tf.placeholder(tf.float32, [None, img_size*img_size, num_channels])
x = tf.reshape(x, shape=[-1, img_size, img_size, num_channels])
# print(x.get_shape())
y = tf.placeholder(tf.float32, [None, n_output_size, num_parts])
y_heat_map = tf.reshape(y, [-1, heat_map_size, heat_map_size, num_parts])
#sets_image = tf.placeholder(tf.float32, [None, n_input, img_size, img_size, num_channels])
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

def g_input_Data():
    # need is an np.array that's
    # (264, 264, 3)
    going_out = np.empty([1, 256, 256, 3])
    for x in range(0,256):
        for y in range(0,256):
            for z in range(0, 3):
                going_out[0,x,y,z] = randint(0,1000)
    return going_out
    
def getImageData():
    print("started the queue")
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    print("read the queue")
    my_img = tf.image.decode_jpeg(value)
    print("decoded the queue")
    t = my_img.eval()
    print("evaluation complete")
    return t.reshape([1, 256, 256, 3])

def get_heat_map_data(body_parts):
    output = np.empty([1,64,64, 7])
    for y in range(0,num_parts):
        output[0, 0:64, 0:64, y] = encode_gauss_map(body_parts[2*y],body_parts[2*y+1])
    print(output.shape)
    return output.reshape([1,64*64, 7])

# in the program what you have is 


# Construct model

print(weights['wc1'])
pred = conv_net(x, weights, biases)

print(pred.get_shape())

# Define loss and optimizer
output = tf.sub(pred, y_heat_map)
output = tf.square(pred, name = None)
print(output.get_shape())

cost = tf.reduce_mean(output)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.initialize_all_variables()

filepaths_o, limb_points_o = read_label_file(train_labels_file, dataset_path)
f_names = tf.placeholder("string", [None])
filename_queue = tf.train.string_input_producer(filepaths_o)


# filename_queue = tf.train.string_input_producer(f_names)
#filename_queue = tf.train.string_input_producer([dataset_path + 'rescale_12-oclock-high-special-edition-00004151.jpg', dataset_path + 'rescale_12-oclock-high-special-edition-00004151.jpg'])
dd = tf.placeholder(tf.float32, [256, 256, 3])
yd = dd

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    #result = sess.run(y ,feed_dict={x: getImageData()})
    #print(result)
    # so you also have to put values into the heatmaps.

    # for y I have to generate a (1, 7, 64, 64) sized tensor
    #filepaths_t, limb_points_t = read_label_file(train_labels_file, dataset_path)


    
    for jt in range(0,3):
        result = sess.run(output ,feed_dict={y: get_heat_map_data(limb_points_o), x:getImageData() })
        print(result)
    #    result = sess.run(cost ,feed_dict={x: getImageData()})
    #    print(result)
    coord.request_stop()
    coord.join(threads)
    #for x in range(0,20):
    # first step is get a list of 20 filepaths and 20* 7 coordinates
    # that can use to put in the Gauss map
    # step 1 is to get the filepaths


    # step 2 is to load these into a
    # step 3 is to put the data from each of these into
    # now have to loop over each file in the queue and put that
    #
    #    for k in range(0, len(filepaths)):
    #   print("started the queue")
    #    reader = tf.WholeFileReader()
    #    key, value = reader.read(filename_queue)
    #    print("read the queue")
    #    my_img = tf.image.decode_jpeg(value)
    #    print("decoded the queue")
    #    t = my_img.eval()
    #    t.reshape([1, 256, 256, 3])
    #    print("evaluation complete")
    #    for y in range(0,num_parts):
    #        print(limb_points[2*y])
    #        curr_target[y, 0:64, 0:64] = encode_gauss_map(limb_points[2*y],limb_points[2*y+1])
    #    print("got the limb data")
    #    print(sess.run(pred, feed_dict={x: t}))
  


    '''
    coord = tf.train.Coordinator()
    enqueue_thread = threading.Thread(target=enqueue, args=[sess])
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    enqueue_thread.isDaemon()
    enqueue_thread.start()

    run_options = tf.RunOptions(timeout_in_ms=4000)
    curr_data_batch, curr_target_batch = sess.run([data_batch, target_batch], options=run_options)
    print(curr_data_batch)

    # shutdown everything to avoid zombies
    sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    coord.join(threads)
    sess.close()
    '''
    
'''
    # Fetch the data from the pipeline and put it where it belongs (into your model)
    for i in range(8):
      run_options = tf.RunOptions(timeout_in_ms=4000)
      batch_x, batch_y = sess.run([data_batch, target_batch], options=run_options)
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
                                      y: mnist.test.labels[:256], keep_prob: 1.}))
      #print(curr_data_batch)
      #print (curr_target_batch)

    # shutdown everything to avoid zombies

    sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    coord.join(threads)
    sess.close()
    
    threads = tf.train.start_queue_runners(coord=coord)
    # here what I want to change is to no longer use the random
    # numbers but instead
    for j in range(0,2):
        print(sess.run(pred, feed_dict={x: getImageData()}))
    coord.request_stop()
    coord.join(threads)

'''

'''

# Initializing the variables
init = tf.initialize_all_variables()https://mail.google.com/mail/u/0/#inbox

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


# start the threads for our FIFOQueue and batch
with tf.Session() as sess:
    enqueue_thread = threading.Thread(target=enqueue, args=[sess])
    enqueue_thread.isDaemon()
    enqueue_thread.start()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Fetch the data from the pipeline and put it where it belongs (into your model)
    for i in range(8):
      run_options = tf.RunOptions(timeout_in_ms=4000)
      batch_x, batch_y = sess.run([data_batch, target_batch], options=run_options)
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
                                      y: mnist.test.labels[:256], keep_prob: 1.}))
      #print(curr_data_batch)
      #print (curr_target_batch)

    # shutdown everything to avoid zombies

    sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    coord.join(threads)
    sess.close()
'''


