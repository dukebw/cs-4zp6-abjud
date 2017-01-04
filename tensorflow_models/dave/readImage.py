# TensorFlow Input Pipelines for Large Data Sets
# ischlag.github.io
# TensorFlow 0.11, 07.11.2016

import tensorflow as tf
import numpy as np
import threading
import math
import sqlite3
#import pandas as pd
import PIL
from PIL import Image
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from random import randint


def getImageData():
    print("started the queue")
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    print("read the queue")
    my_img = tf.image.decode_jpeg(value)
    print("decoded the queue")
    t = my_img.eval()
    print("evaluation complete")
    return t #.set_shape([256, 256, 3])

dataset_path = "/home/david/FLIC-full/images/"
# so it's the process of converting from

init = tf.initialize_all_variables()
f_names = tf.placeholder("string", [None])
# filename_queue = tf.train.string_input_producer(f_names)
filename_queue = tf.train.string_input_producer([dataset_path + 'rescale_12-oclock-high-special-edition-00004151.jpg'])
x = tf.placeholder("float", [256, 256, 3])
y = x



with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    result = sess.run(y ,feed_dict={x: getImageData()})
    print(result)
    coord.request_stop()
    coord.join(threads)
