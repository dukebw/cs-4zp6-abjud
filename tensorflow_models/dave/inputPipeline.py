from __future__ import print_function

import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

# all you have to do is generate a tensor of inputs and a tensor
# of desired outputs

# feed_dict_train = {x: x_batch, y_true: y_true_batch}
# then you say sess.run(optimizer, feed_dict={input1:[7.], input2:[2.]})
# define optimizer with tf.train.AdamOptimizer(learningRate).minimize(cost)
# cost = tf.reduce_mean(cross_entropy)
# cross_entropy(logits = layerfc2, labels=y_true)

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
train_labels_file = "pose_DB.sqlite"


# probably want to change most of this
test_set_size = 5
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CHANNELS = 3


# Creating a pipeline step 1
# Load the label data which contains lists
# of input files    

def read_label_file(file):
    numberOfFiles = 20
    #f = open(file, 'r')
    conn = sqlite3.connect(file)
    c = conn.cursor()
    c.execute("SELECT filename, left_x_shoulder, left_y_shoulder FROM pose_train_table WHERE file_key \
            < 20")#, (numberOfFiles))
    conn.commit()
    #row = c.fetchone()
    filepaths = []
    labels = [] # these labels will be 14 numbers but for now you could just use the shoulder data
    fetch = c.fetchmany
    rows = fetch(numberOfFiles)
    for row in rows:
        print ("it's going")
        filepaths.append(row[0])
        labels.append(encode_label(row[1], row[2]))
    return filepaths, labels

def encode_label(label1, label2): # here I should create a 64x64 sized Gaussian
    return label1, label2


# here call both of the above methods
train_filepaths, train_labels = read_label_file(train_labels_file) #dataset_path + train_labels_file)
test_filepaths, test_labels = read_label_file(train_labels_file)

for x in range(1,19):
    print (train_filepaths[x])

print (train_filepaths[10])

# what it should do is grab 
# all_images = ops.convert_to_tensor(train_filepaths, dtype=dtypes.string)
# all_labels = ops.convert_to_tensor(train_labels, dtype=dtypes.float16)
# I don't see any reason to convert the files to tensors.
# now I have a list of files I try to convert each of those
# jpeg images into tensors.

x = tf.placeholder(tf.float32, shape = [None, img_size_flat], name='x')

# reshape
x_image= tf.reshape(-1, img_size, img_size, num_channels)

y_true = tf.placeholder(tf.float32, shape=[None, 64, 64], name='y_true')
# interesting y_true_cls = tf.argmax(y_true, dimension=1)
# think this means take the max value and set it to something

# what I should do is define simply a tensor that maps from 256 x 256
# down to 64 x 64

# but what exactly is x_batch? Is it a list of tensors or something?


# they say a feed dictionary lets you inject data into a graph
# print(classifier.eval(feed_dict={input: my_python_preprocessing_fn()}))

# a placeholder acts solely as the target of feeds.
# say there's an example of this in fully connected feeds

# a typical pipeline for reading records from files consists of
#1. list of filenames (check)
#2. Optional filename shuffling
#3. Optional Epoch limit 
#4. Filename queue (check)
#5. A reader for the file format
#6. A decorder for the record read by the reader
#7. Optional pre-processing
#8. Example Queue

# pass the filename queue to the reader's read method
# it outputs a key identifying the file and record - and a scalar string value
# then you use the decoder and conversion ops to decode this string into a tensor
# what I might have to do is convert each file into a 64x64 jpeg which I can compare
# the output too.

# if file_name_queue is a queue of filenames
# then
reader = tf.WholeFileReader()

key, value = reader.read(filename_queue)

my_img = tf.image.decode_jpg(value)
