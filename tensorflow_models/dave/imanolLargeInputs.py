# TensorFlow Input Pipelines for Large Data Sets
# ischlag.github.io
# TensorFlow 0.11, 07.11.2016

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
#
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
n_input = img_size*img_size # MNIST data input (img shape: 28*28)
n_output_size = heat_map_size * heat_map_size
#n_classes = 10 # MNIST total classes (0-9 digits)
#dropout = 0.75 # Dropout, probability to keep units


# are used to feed data into our queue

queue_input_data = tf.placeholder(tf.float32, shape=[num_in_batch, data_size_input])
queue_input_target = tf.placeholder(tf.float32, shape=[num_in_batch, data_size_label])
queue = tf.FIFOQueue(capacity=200, dtypes=[tf.float32, tf.float32], shapes=[[data_size_input], [data_size_label]])
enqueue_op = queue.enqueue_many([queue_input_data, queue_input_target])
dequeue_op = queue.dequeue()
# tensorflow recommendation:
# capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=15, capacity=40)
# use this to shuffle batches:
# data_batch, target_batch = tf.train.shuffle_batch(dequeue_op, batch_size=15, capacity=40, min_after_dequeue=5)


def read_label_file(file, path):
    numberOfFiles = 20
    #f = open(file, 'r')
    conn = sqlite3.connect(file)
    c = conn.cursor()
    # note that can't adjust the number of files.Have to return multipes of 20 -- you can see why if
    # you look at the line below where the 20 is encoded into the SELECT statement.
    c.execute("SELECT filename, right_x_shoulder, right_y_shoulder, left_x_shoulder, left_y_shoulder, right_x_wrist, right_y_wrist, left_x_wrist, left_y_wrist, left_x_elbow, left_y_elbow, right_x_elbow, right_y_elbow, x_head, y_head FROM pose_train_table WHERE file_key \
            < 20") #, (numberOfFiles))
    conn.commit()
    #row = c.fetchone()    
    filepaths = []
    part_list = []
    #labels = [] # these labels will be 14 numbers but for now you could just use the shoulder data
    fetch = c.fetchmany
    rows = fetch(numberOfFiles)
    for row in rows:
        print ("it's going")
        filepaths.append( path +'rescale_'+row[0].strip())
        rescale_the_image(path + row[0].strip(),path + 'rescale_'+row[0].strip())
        for part_num in range(0, num_parts):
            #print(part_num)
            part_list.append(row[2*part_num+1])
            part_list.append(row[2*part_num+2])
    return filepaths, part_list

def encode_gauss_map(x_avg, y_avg): # here I should create a 64x64 sized array
    labels = []
    for x in range(0, heat_map_size):
        for y in range(0, heat_map_size):
            value = 2*gauss_deviation**2
            value = ((x - x_avg)**2 + (y - y_avg)**2)/value
            value = norm_factor * math.e **(-value)
            labels.append(value)
    return np.array(labels).reshape(64,64)

def enqueue(sess):
  """ Iterates over our data puts small junks into our queue."""
  under = 0
  max = 300 #len(raw_data)
  print(max)
  print(under+20)
  image_data_list = []
  while under < (max -20):
    print("starting to write into queue")
    upper = under + 20
    print("try to enqueue ", under, " to ", upper)
    if upper <= max:
        curr_target_array = np.empty([20, num_parts, heat_map_size, heat_map_size])
        image_data = np.empty([20, img_size, img_size, num_channels])
        #for x in range(0,20):
        # first step is get a list of 20 filepaths and 20* 7 coordinates
        # that can use to put in the Gauss map
        filepaths, curr_target = read_label_file(train_labels_file, dataset_path)
        for x in range(0, len(filepaths)):
            print(filepaths[x])
            image_data = retrieveData(filepaths[x])
            image_data_list.append(image_data.set_shape([img_size, img_size, num_channels]))
            for y in range(0,num_parts):
                curr_target_array[x,y,0:64,0:64] = encode_gauss_map(curr_target[2*y],curr_target[2*y+1])
            if x < 1:
                print(sess.run(image_data))
            # here have to get the data from the images
            # want a tensor 20, 256, 256
            #curr_data = raw_data[under:upper]
            # get a list of filenames
            # done this and then you
        print("about to concat")
        image_data = tf.concat(0,image_data_list)
        print("about to concat")
        sess.run(enqueue_op, feed_dict={queue_input_data: image_data,
                                    queue_input_target: curr_target_array})
        print("done a loop")
    else:
        #break
        rest = upper - max
        curr_data = np.concatenate((raw_data[under:max], raw_data[0:rest]))
        curr_target = np.concatenate((raw_target[under:max], raw_target[0:rest]))
        under = rest
        # only do this a couple times to see if 
    under = upper
    #sess.run(dequeue_op)
    print("added to the queue")
    #if sess.should_stop():
    #    print("Trying to BBBBBREAK")
    #    break
  print("finished enqueueing")


def retrieveData(file):
    reader = tf.WholeFileReader()
    key, value = reader.read(file)
    #file_content = tf.read_file(file)
    train_image = tf.image.decode_jpeg(file_content, channels=num_channels)
    train_image.set_shape([img_size, img_size, num_channels])
    print(train_image.get_shape())
    return train_image

def rescale_the_image(filePath, newName):
    basewidth = 256
    img = Image.open(filePath) #'/home/david/FLIC-full/images/2-fast-2-furious-00003571.jpg')
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth, basewidth), PIL.Image.ANTIALIAS)
    img.save(newName) #'resized_image.jpg')

'''
def enqueue(sess):
  """ Iterates over our data puts small junks into our queue."""
  under = 0
  max = 300 #len(raw_data)
  print(max)
  print(under+20)
  while under < (max -20):
    print("starting to write into queue")
    upper = under + 20
    print("try to enqueue ", under, " to ", upper)
    if upper <= max:
        curr_target = np.empty([20,25])
        for x in range(0,20):
            curr_target[x,:] = np.array(encode_gauss_map(1,1))
            curr_data = raw_data[under:upper]
        sess.run(enqueue_op, feed_dict={queue_input_data: curr_data,
                                    queue_input_target: curr_target})
        print(curr_target.shape)
    else:
        #break
        rest = upper - max
        curr_data = np.concatenate((raw_data[under:max], raw_data[0:rest]))
        curr_target = np.concatenate((raw_target[under:max], raw_target[0:rest]))
        under = rest
        # only do this a couple times to see if 
    under = upper
    #sess.run(dequeue_op)
    print("added to the queue")
    #if sess.should_stop():
    #    print("Trying to BBBBBREAK")
    #    break
  print("finished enqueueing")
'''
# start the threads for our FIFOQueue and batch
sess = tf.Session()
enqueue_thread = threading.Thread(target=enqueue, args=[sess])
enqueue_thread.isDaemon()
enqueue_thread.start()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# Fetch the data from the pipeline and put it where it belongs (into your model)

for i in range(8):
  run_options = tf.RunOptions(timeout_in_ms=4000)
  #curr_data_batch, curr_target_batch = sess.run([data_batch, target_batch], options=run_options)
  #print(curr_data_batch)
  #print (curr_target_batch)

# shutdown everything to avoid zombies

sess.run(queue.close(cancel_pending_enqueues=True))
coord.request_stop()
coord.join(threads)
sess.close()


