"""This module implements the TensorFlow HTTP server, which is a standalone
server that is able to handle HTTP requests to do TensorFlow operations.
"""
import re
import base64
import ssl
import urllib
import http.server
import requests
import numpy as np
import tensorflow as tf
from human_pose_model.networks.vgg_bulat import two_vgg_16s_cascade

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

RESTORE_PATH = '/mnt/data/datasets/MPII_HumanPose/logs/vgg_bulat/both_nets_xentropy_regression/23'
IMAGE_DIM = 380

def _get_image_joint_predictions(image,
                                 session,
                                 image_bytes_feed,
                                 logits_tensor):
    """
    """
    feed_dict = {
        image_bytes_feed: image
    }
    logits = session.run(fetches=logits_tensor, feed_dict=feed_dict)

    batch_size = logits.shape[0]
    num_joints = logits.shape[-1]
    x_predicted_joints = np.empty((batch_size, num_joints))
    y_predicted_joints = np.empty((batch_size, num_joints))

    # {["r_ankle": [0.5, 0.5], "r_knee": [1.0, 2.0], ...],
    #  ["r_ankle": [0.25, 0.2], "r_knee": [0.1, 0.5], ...]}
    joint_predictions_json = '['
    for batch_index in range(batch_size):
        joint_predictions_json += '{'
        for joint_index in range(num_joints):
            joint_heatmap = logits[batch_index, ..., joint_index]
            xy_max_confidence = np.unravel_index(
                joint_heatmap.argmax(), joint_heatmap.shape)

            y_joint = xy_max_confidence[0]/IMAGE_DIM - 0.5
            x_joint = xy_max_confidence[1]/IMAGE_DIM - 0.5

            y_predicted_joints[batch_index, joint_index] = y_joint
            x_predicted_joints[batch_index, joint_index] = x_joint

            joint_predictions_json += ('"{}": [{}, {}]'
                                       .format(JOINT_NAMES_NO_SPACE[joint_index], str(x_joint), str(y_joint)))
            if joint_index < (num_joints - 1):
                joint_predictions_json += ', '

        joint_predictions_json += '}'
        if batch_index < (batch_size - 1):
            joint_predictions_json += ', '
    joint_predictions_json += ']'

    return joint_predictions_json


def TFHttpRequestHandlerFactory(session, image_bytes_feed, logits_tensor):
    class TFHttpRequestHandler(http.server.BaseHTTPRequestHandler):
        """Defines handlers for specific HTTP request codes."""
        def __init__(self, *args, **kwargs):
            super(TFHttpRequestHandler, self).__init__(*args, **kwargs)

        def _respond_with_joints(self, image):
            """
            """
            joint_predictions_json = _get_image_joint_predictions(
                image,
                session,
                image_bytes_feed,
                logits_tensor)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', 'https://brendanduke.ca')
            self.end_headers()

            self.wfile.write(joint_predictions_json.encode('utf8'))

        def do_GET(self):
            """
            """
            image_url = self.requestline.split()[1]
            image_url = urllib.parse.unquote(image_url)
            image_url = re.match('/\?image_url=(.*)', image_url)
            assert image_url is not None

            image_url = image_url.groups()[0]

            image_request = requests.get(image_url)

            self._respond_with_joints(image_request.content)

        def do_POST(self):
            """
            """
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            self._respond_with_joints(base64.b64decode(post_data))

    return TFHttpRequestHandler


def run():
    """Starts a server that will handle  HTTP requests to use TensorFlow."""
    with tf.Graph().as_default():
        image_bytes_feed = tf.placeholder(dtype=tf.string)

        decoded_image = tf.image.decode_jpeg(contents=image_bytes_feed)
        decoded_image = tf.image.convert_image_dtype(image=decoded_image,
                                                     dtype=tf.float32)

        image_shape = tf.shape(input=decoded_image)
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

        logits, _ = two_vgg_16s_cascade(normalized_image, 16, False, False)

        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=RESTORE_PATH)
        assert latest_checkpoint is not None

        restorer = tf.train.Saver(var_list=tf.global_variables())
        restorer.restore(sess=session, save_path=latest_checkpoint)

        request_handler = TFHttpRequestHandlerFactory(session,
                                                      image_bytes_feed,
                                                      logits)
        server_address = ('localhost', 8765)
        httpd = http.server.HTTPServer(server_address, request_handler)
        # openssl req -new -x509 -keyout server.pem -out server.pem -days 365 -nodes
        httpd.socket = ssl.wrap_socket(httpd.socket,
                                       keyfile='./domain.key',
                                       certfile='./signed.crt',
                                       server_side=True)
        httpd.serve_forever()


if __name__ == "__main__":
    run()
