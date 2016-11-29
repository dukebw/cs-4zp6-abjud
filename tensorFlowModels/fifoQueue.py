import sqlite3
import tensorflow as tf
import numpy as np
import random
import math
import PIL
from PIL import Image
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from random import randint


'''
q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([0,0,0],))

x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])

with tf.Session() as sess:

    print(init.run())
    print(sess.run(q_inc))
    #print(sess.run(q.dequeue()))
    #print(sess.run(q.dequeue()))
    #print(sess.run(q.dequeue()))
    q_inc.run()
    #print(sess.run(q.dequeue()))
    
    q_inc.run()
    q_inc.run()
'''
dataset_path = "/home/david/FLIC-full/images/"

# maybe I should have a separate database with the test labels?
test_labels_file = "pose_DB.sqlite"

train_labels_file = "pose_DB.sqlite"


# probably want to change most of this

# note all image sizes from the FLIC database have to be reset

test_set_size = 20
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
    
# here just grab one entry at random
def read_label_file(file):
    rand_num = randint(1,10000)
    #numberOfFiles = 20
    #f = open(file, 'r')
    conn = sqlite3.connect(file)
    c = conn.cursor()
    c.execute("SELECT filename, left_x_shoulder, left_y_shoulder FROM pose_train_table WHERE file_key \
            =" + str(rand_num)) #, (numberOfFiles))
    conn.commit()
    #row = c.fetchone()
    #filepaths = []
    #labels = [] # these labels will be 14 numbers but for now you could just use the shoulder data
    #fetch = c.fetchmany
    rows = c.fetchall() #s = fetch(numberOfFiles)
    for row in rows:
        print ("it's going")
        #filepaths.append( 'rescale_'+row[0].strip())
        #rescale_the_image('/home/david/FLIC-full/images/' + row[0].strip(),'/home/david/FLIC-full/images/' + 'rescale_'+row[0].strip())
        #labels.append(encode_gauss_map(row[1], row[2]))
        filepaths = 'rescale_'+row[0].strip()
        labels = row[1]
    return filepaths, labels

def encode_label(label1, label2): # here I should create a 64x64 sized Gaussian
    return label1, label2



# here need to put the data into a numby array
# then a feed can be used

def encode_gauss_map(x_avg, y_avg): # here I should create a 64x64 sized array
    labels = []
    for x in range(1, 65):
        for y in range(1, 65):
            value = 2*gauss_deviation**2
            value = ((x - x_avg)**2 + (y - y_avg)**2)/value
            value = norm_factor * math.e **(-value)
            labels.append(value)
    return labels



# now try to put a tensor in the queue

def rddude():
    # Conv2D wrapper, with bias and relu activation
    return randint(0,9)

# constructor has these issues:
# 1. capacity, dtypes, shapes
q = tf.FIFOQueue(3, dtypes=[tf.string, tf.float32], shapes=[[1],[64*64]])
#init = q.enqueue_many(([0,0,0],))
'''
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])
'''
# now what I can try to do is get data from the database and 

with tf.Session() as sess:
    

    # put this in a np array
    # how you can work on this is just put the data into
    # these np_arrays and then
    # sess.run(model, feed_dict={X: trY, Y: trY})
    # where trY and trX are 
    a = np.array(encode_gauss_map(0,0))
    
    # put this a into a queue
    # you can only enqueue a tensor
    q.enqueue("right on", a)
    #print(encode_gauss_map(0,0))
    #print(read_label_file("pose_DB.sqlite"))
    #print(read_label_file("pose_DB.sqlite"))
    #print(read_label_file("pose_DB.sqlite"))
    sess.close()
    
'''
    print(init.run())
    print(sess.run(q_inc))
    #print(sess.run(q.dequeue()))
    #print(sess.run(q.dequeue()))
    #print(sess.run(q.dequeue()))
    q_inc.run()
    #print(sess.run(q.dequeue()))
    
    q_inc.run()
    q_inc.run()
'''
