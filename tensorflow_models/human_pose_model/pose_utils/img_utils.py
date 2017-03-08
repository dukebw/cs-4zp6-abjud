"""This module contains image manipulation functions, such as for drawing and
showing images, that can be used for debugging.
"""
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

def draw_binary_maps(image_tensor, binary_map_tensor, num_to_draw):
    """Draws binary maps on `num_to_draw` images, and displays them."""
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        for _ in range(num_to_draw):
            image, binary_map = session.run([tf.image.convert_image_dtype(image=image_tensor, dtype=tf.uint8),
                                             binary_map_tensor])

            for joint_index in range(binary_map.shape[-1]):
                if joint_index in [6, 7, 8, 9]:
                    colour_index = 2
                if joint_index in [0, 1, 2, 10, 11, 12]:
                    colour_index = 1
                else:
                    colour_index = 0

                image[..., colour_index] = np.clip(
                    image[..., colour_index] + 4*255*binary_map[..., joint_index],
                    0,
                    255)

            pil_image = Image.fromarray(image)
            pil_image.show()

        coord.request_stop()
        coord.join(threads=threads)


def draw_logits(image_batch, logits):
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

            cv2.imwrite('images/img{}_{}.jpg'.format(image_index, joint_index),
                        heatmap_on_image)
