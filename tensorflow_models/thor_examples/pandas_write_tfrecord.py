import numpy as np
import os
import os.path
import tensorflow as tf
from tqdm import tqdm
from natsort import natsorted
import random
import time
import re
from pprint import pprint

""" dataframe example:
     filename                                          015601864.jpg
     head_rect                          [627.0, 100.0, 706.0, 198.0]
     is_visible    {'0': 1, '2': 1, '8': 0, '14': 1, '15': 1, '13...
     joint_pos     {'0': [620.0, 394.0], '2': [573.0, 185.0], '8'...
     train                                                         1
     Name: 0, dtype: object
"""
print('Reading mpii_data as a Pandas DataFrame')
mpii_data = './data/mpii/MPII_HumanPose.h5'
"""
 Walks through the path and obtains a list of .jpg files
 The list of files is used to initialize a tf.train.string_input_producer
 Input: Path which contains our images
 Output: tf.train.string_input_producer
"""
def read_filelist(path):
    fileList = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(path)
        for f in files if f.endswith('.jpg')]
    fileList= natsorted(fileList)
    print "No of files: %i" % len(fileList)
    files =  tf.train.string_input_producer(fileList,shuffle=False)
    return files

"""
 Utility function
 Input: value for int64 list
 Output: tf.train.Feature
"""
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

"""
 Utility function
 Input: value for bytes list
 Output: tf.train.Feature
"""
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

"""
 Takes in a Pandas dataframe, gets the list of filenames and decodes the files
 into tensors.
 Input: Pandas DataFrame
 Output: my_img in tensor format
"""
def convert2tensor(mpii_data)
    # init file reader
    reader = tf.WholeFileReader()
    # get mpii files to fetch as a list and pass to reader
    files = mpii_data['filename'].tolist()
    # definition of reader?
    _, value = reader.read(files)
    my_img = tf.image.decode_jpeg(value) #reads images into tensor
    return my_img
"""
 Takes a Pandas dataframe and converts it to a TFRecord
 Input: Pandas dataframe
 Output: TFRecord
"""
def serialize(mpii_data)
    my_img = convert2tensor(mpii_data)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        filename = os.path.join('mpii_HumanPose.tfrecords')
        print('Writing', filename)
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        writer = tf.python_io.TFRecordWriter(filename)

        # Not Finished, need to understand how to augment labels

        seq_idx= xrange(len(labels))
        train_idx = seq_idx[500:-1000]
        val_idx = seq_idx[500:-1000]
        test_idx = seq_idx[:500]

        for index in tqdm(range(seq_length)):  #(11648)
            image = my_img.eval() 
            image_raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

    coord.request_stop()
    coord.join(threads)
