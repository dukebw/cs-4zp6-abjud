import sqlite3
import tensorflow as tf
import random
import math
import PIL
from PIL import Image
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

dataset_path = "/home/david/FLIC-full/images/"

# maybe I should have a separate database with the test labels?
test_labels_file = "pose_DB.sqlite"

train_labels_file = "pose_DB.sqlite"


# probably want to change most of this

# note all image sizes from the FLIC database have to be reset

test_set_size = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CHANNELS = 3
BATCH_SIZE = 5

# parameters for the Gaussian
gauss_deviation = 10
norm_factor = 2*math.pi*gauss_deviation*gauss_deviation
norm_factor = 1/norm_factor

# try to resize an image
'''
basewidth = 300
img = Image.open('/home/david/FLIC-full/images/2-fast-2-furious-00003571.jpg')
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((256, 256), PIL.Image.ANTIALIAS)
img.save('resized_image.jpg')
'''
# the pain is that these transformations also have to be applied to the data



# Creating a pipeline step 1
# Load the label data which contains lists
# of input files    


# one problem with this is that the pose coordinates of the input has to be
# re-scaled 


# also the encode_gauss_map returns a list and then you are appending lists together
# which might cause problems

def rescale_the_image(filePath, newName):
    basewidth = 256
    img = Image.open(filePath) #'/home/david/FLIC-full/images/2-fast-2-furious-00003571.jpg')
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth, basewidth), PIL.Image.ANTIALIAS)
    img.save(newName) #'resized_image.jpg')
    

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
        filepaths.append( 'rescale_'+row[0].strip())
        rescale_the_image('/home/david/FLIC-full/images/' + row[0].strip(),'/home/david/FLIC-full/images/' + 'rescale_'+row[0].strip())
        labels.append(encode_gauss_map(row[1], row[2]))
    return filepaths, labels

def encode_label(label1, label2): # here I should create a 64x64 sized Gaussian
    return label1, label2


def encode_gauss_map(x_avg, y_avg): # here I should create a 64x64 sized array
    labels = []
    for x in range(1, 65):
        for y in range(1, 65):
            value = 2*gauss_deviation**2
            value = ((x - x_avg)**2 + (y - y_avg)**2)/value
            value = norm_factor * math.e **(-value)
            labels.append(value)
    return labels

#print(encode_gauss_map(0,0))


# here call both of the above methods
train_filepaths, train_labels = read_label_file(train_labels_file) #dataset_path + train_labels_file)
train_filepaths = [dataset_path + fp for fp in train_filepaths]

#test_filepaths, test_labels = read_label_file(train_labels_file)

#for x in range(1,19):
#    print (train_filepaths[x])
#    print (train_labels[x])

#print (train_filepaths[10])

# up to here works fine

# convert to tensors
all_images = ops.convert_to_tensor(train_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(train_labels, dtype=dtypes.float16)





#print(all_images)
#print(all_labels)

# create a partition vector

# what exactly is a partition vector?


# this might be a bit weird because the labels contain multiple values
# per data point
# partitions = [0] * (len(train_filepaths) - test_set_size)
# partitions[:test_set_size] = [1] * test_set_size # here add the test data to the partition
lengt = (len(train_filepaths)-0)
print(lengt)
partitions = [0] * lengt
#partitions[:5] = [1] * 5
random.shuffle(partitions)

# partition our data into a test and train set according to our partition vector
train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)

# I think this might cause errors because the label data is longer
# maybe should've created a list of pairs?
train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)


# create input queues
train_input_queue = tf.train.slice_input_producer(
                                    [train_images, train_labels],
                                    shuffle=False)
test_input_queue = tf.train.slice_input_producer(
                                    [test_images, test_labels],
                                    shuffle=False)

# process path and string tensor into an image and a label

# this is strange I should 
file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
train_label = train_input_queue[1]

file_content = tf.read_file(test_input_queue[0])
test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
test_label = test_input_queue[1]

# define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


# collect batches of images before processing
train_image_batch, train_label_batch = tf.train.batch(
                                    [train_image, train_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )
test_image_batch, test_label_batch = tf.train.batch(
                                    [test_image, test_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )



with tf.Session() as sess:

    
  # session 
  # initialize the variables
  sess.run(tf.initialize_all_variables())
  
  # initialize the queue threads to start to shovel data
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print ("from the train set:")
  for i in range(20):
    print (sess.run(train_label_batch))
    print ("image set")
    print (sess.run(train_image_batch))

#  print ("from the test set:")
#  for i in range(10):
#    print (sess.run(test_label_batch))

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  sess.close()


