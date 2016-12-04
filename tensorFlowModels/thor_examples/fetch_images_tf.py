import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from PIL import Image
import tensorflow as tf

data = 'data/mpii/'

def get_images():
    img_df = pd.read_hdf(data + 'mpii_train.hdf')
    return img_df

def get_img_fname(img_df):
    img_fname = img_df['filename'][0]
    #img = mpimg.imread(data + 'images/' + img_fname)
    return img_fname

def get_img(img_fname):
    filename_queue = tf.train.string_input_producer([data + 'images/' + img_fname]) #  list of files to read
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    my_img = tf.image.decode_jpeg(value) # use png or jpeg decoder based on your files.
    return my_img

def fetch_img_tensor():
    # Fetch data
    img_df = get_images() # Gets images from pandas
    img_fname = get_img_fname(img_df)
    my_img = get_img(img_fname)
    return my_img

if __name__ == "__main__":
    # Note that my_img can just as well be an array of images since this is a
    # tensor
    my_img = fetch_img_tensor()
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        ''' Build Graph'''
        
        ''' Evaluate Tensor'''
        for i in range(1): #length of your filename list
            image = my_img.eval() #here is your image Tensor :) 
        print(image.shape)
        Image.fromarray(np.asarray(image)).show()
        coord.request_stop()
        coord.join(threads)
