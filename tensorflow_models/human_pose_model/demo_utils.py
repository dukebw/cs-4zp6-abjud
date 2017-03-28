"""This module implements the TensorFlow HTTP server, which is a standalone
server that is able to handle HTTP requests to do TensorFlow operations.
"""
#import cv2 #We have to import cv2 before import tensorflow!!
import tensorflow as tf
import os
import re
import base64
import ssl
import urllib3
#import http.server
#import requests
import numpy as np
#from matplotlib import pylab
#import imageio
from PIL import Image
from networks.vgg_bulat import two_vgg_16s_cascade
from networks.resnet_bulat import resnet_50_detector
from networks.vgg_bulat import vgg_16_bn_relu
# Restrict tensorflow to only use the first GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

JOINT_NAMES_NO_SPACE = ['r_ankle',
                        'r_knee',
                        'r_hip',
                        'l_hip',
                        'l_knee',
                        'l_ankle',
                        'pelvis',
                        'thorax',
                        'upper_neck',
                        'head_top',
                        'r_wrist',
                        'r_elbow',
                        'r_shoulder',
                        'l_shoulder',
                        'l_elbow',
                        'l_wrist']

resnet50_PATH = '/media/ubuntu/SD/data/resnet50'
both_nets_PATH = '/media/ubuntu/SD/data/both_nets'
detector_PATH = '/media/ubuntu/SD/data/detector'

IMAGE_DIM = 260
RESTORE_PATH = detector_PATH

def get_joint_position_inference_graph(image_bytes_feed, get_logits_function):
    """This function sets up a computation graph that will decode from JPEG the
    input placeholder image `image_bytes_feed`, pad and resize it to shape
    [IMAGE_DIM, IMAGE_DIM], then run joint inference on the image using the
    "Two VGG-16s cascade" model.
    """
    decoded_image = tf.image.decode_jpeg(contents=image_bytes_feed)
    decoded_image = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)

    image_shape = tf.shape(input=decoded_image)
    print(image_shape)
    height = image_shape[0]
    width = image_shape[1]
    pad_dim = tf.cast(tf.maximum(height, width), tf.int32)
    offset_height = tf.cast(tf.cast((pad_dim - height), tf.float32)/2.0, tf.int32)
    offset_width = tf.cast(tf.cast((pad_dim - width), tf.float32)/2.0, tf.int32)
    padded_image = tf.image.pad_to_bounding_box(image=decoded_image,
                                                offset_height=offset_height,
                                                offset_width=offset_width,
                                                target_height=pad_dim,
                                                target_width=pad_dim)

    resized_image = tf.image.resize_images(images=padded_image,
                                           size=[IMAGE_DIM, IMAGE_DIM])

    normalized_image = tf.subtract(x=resized_image, y=0.5)
    normalized_image = tf.multiply(x=normalized_image, y=2.0)

    normalized_image = tf.reshape(tensor=normalized_image,
                                  shape=[IMAGE_DIM, IMAGE_DIM, 3])

    normalized_image = tf.expand_dims(input=normalized_image, axis=0)

    logits, _ = get_logits_function(normalized_image, 16, False, False)

    return logits


def draw_logits(image, logits):
    """Draws heatmaps of joint logits on each image in batch, and saves one
    image per joint to the images/ directory.
    """
    for image_index in range(logits.shape[0]):
        _, threshold = cv2.threshold(logits[image_index, ...],
                                     5,
                                     255,
                                     cv2.THRESH_BINARY)

        for joint_index in range(logits.shape[-1]):
            dist_transform = cv2.distanceTransform(threshold[..., joint_index].astype(np.uint8), cv2.DIST_L1, 3)
            dist_transform = cv2.convertScaleAbs(dist_transform, dist_transform, 255.0/np.max(dist_transform))
            heatmap = cv2.applyColorMap(dist_transform, cv2.COLORMAP_JET)

            alpha = 0.5
            heatmap_on_image = cv2.addWeighted(image_batch[image_index],
                                               alpha,
                                               heatmap,
                                               1.0 - alpha,
                                               0.0)
            return heatmap_on_image


class ImageHandler(object):
    # What purpose does this class have?
    # This class serves to handle an image and a pose estimate with associated information from CNN
    # Later we will generalize this object to handle a stream of data
    def __init__(self,filename, get_logits_function):
        self.filename = filename
        self.image = self._read()
        self.get_logits_function = get_logits_function
        self.init_session()

    def _read(self):
        with open(self.filename, "rb") as imageFile:
            f = imageFile.read()
            #b = bytearray(f)
            return f

    def init_session(self):
        # Note: This function needs to be called after make_graph
        with tf.Graph().as_default():
            self.image_bytes_feed = tf.placeholder(dtype=tf.string)
            self.logits = get_joint_position_inference_graph(self.image_bytes_feed, self.get_logits_function)
            shape = self.logits.get_shape()
            #merged_logits = tf.reshape(tf.reduce_max(self.logits, 3),[1, IMAGE_DIM, IMAGE_DIM, 1])
            #self.pose = tf.cast(merged_logits, tf.float32)
            self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=RESTORE_PATH)
            assert latest_checkpoint is not None

            restorer = tf.train.Saver(var_list=tf.global_variables())
            restorer.restore(sess=self.session, save_path=latest_checkpoint)

    def display(self):
        fig = pylab.figure()
        fig.suptitle('image')
        pylab.imshow(self.image)
        pylab.show()

    def get_pose(self):
        print(type(self.image))
        feed_dict = {self.image_bytes_feed: self.image}
        logits = self.session.run(fetches=[self.logits], feed_dict=feed_dict)
        _, threshold = cv2.threshold(logits[0,...],5,255,cv2.THRESH_BINARY)
        pose_heatmaps = []
        for joint_index in range(Person.NUM_JOINTS):
            dist_transform = cv2.distanceTransform(threshold[..., joint_index].astype(np.uint8), cv2.DIST_L1, 3)
            dist_transform = cv2.convertScaleAbs(dist_transform, dist_transform, 255.0/np.max(dist_transform))
            heatmap = cv2.applyColorMap(dist_transform, cv2.COLORMAP_JET)
            alpha = 0.5
            heatmap_on_image = cv2.addWeighted(image_batch[image_index],alpha,heatmap,1.0 - alpha,0.0)
            pose_maps.append(heatmap_on_image)
        return pose, pose_maps


    def get_feature_maps():
        return self.global_detector_bottleneck, self.global_skiplayers

    def get_nearest_neighbour():
        pass

    def extrap():
        pass

    def interpolate():
        pass

    def deepdream():
        pass


class VideoHandler(object):
		# Maybe make VideoHandler be a list of ImageHandlers?
    def __init__(self, filename):
        self.filename = filename
        self.io_video = imageio.get_reader(filename, 'ffmpeg')

    def get_random_frame(self):
        num = np.random.choice(np.arange(len(self.io_video())))
        return self.io_video.get_data(num)

    def display_random_frame(self):
        fig = pylab.figure()
        fig.suptitle('image #{}'.format(num))
        pylab.imshow(self.get_random_frame())


if __name__ == "__main__":
		run()
		filename = 'messi_juggling.mp4'
		vid = imageio.get_reader(filename,'ffmpeg')
		num_frames = vid._meta['nframes']
		num = np.random_choice(np.arange(num_frames))
		random_image_from_video = vid.get_data(num)
		fig = pylab.figure()
		fig.suptitle('image #{}'.format(num),fontsize=20)
		pylab.imshow(random_image_from_video)
		pylab.show()

