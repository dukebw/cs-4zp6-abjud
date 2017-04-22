"""This module implements the TensorFlow HTTP server, which is a standalone
server that is able to handle HTTP requests to do TensorFlow operations.
"""
import re
import json
import base64
import threading
import urllib
import http.server
import ssl
import requests
import numpy as np
import cv2
import imageio
import tensorflow as tf
import tensorflow.contrib.slim as slim
from human_pose_model.networks import resnet_bulat
from human_pose_model.pose_utils.timethis import timethis

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

RESTORE_PATH = '/mnt/data/datasets/MPII_HumanPose/logs/resnet_brendan/regressor/8'
IMAGE_DIM = 384
BATCH_SIZE = 16

@timethis
def _get_image_joint_predictions(image,
                                 session,
                                 image_bytes_feed,
                                 logits_tensor):
    """This function does joint position inference on a single image, and
    returns the resultant joint predictions in a JSON string.

    The returned JSON string has the following format:

        [{"r_ankle": [0.5, 0.5], "r_knee": [1.0, 2.0], ...},
         {"r_ankle": [0.25, 0.2], "r_knee": [0.1, 0.5], ...}]

    I.e. it is a JSON array of dictionaries, where the keys are names of joints
    from `JOINT_NAMES_NO_SPACE`, and the values are two-element arrays
    containing the [x, y] coordinates of the joint prediction.

    These [x, y] coordinates are in a space where the range [-0.5, 0.5]
    represent the range, in the padded image, from the far left to the far
    right in the case of x, and from the top to the bottom in the case of y.

    Args:
        image: JPEG image in raw bytes format.
        session: TF session to run inference in.
        image_bytes_feed: Placeholder tensor to feed image into.
        logits_tensor: Output logits tensor, corresponding to heatmaps of joint
            positions inferred by the network.

    Returns:
        JSON string of joint positions, as described above.
    """
    logits = session.run(fetches=logits_tensor,
                         feed_dict={image_bytes_feed: image})

    batch_size = logits.shape[0]
    num_joints = logits.shape[-1]
    x_predicted_joints = np.empty((batch_size, num_joints))
    y_predicted_joints = np.empty((batch_size, num_joints))

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


@timethis
def _get_heatmaps_for_batch(frames,
                            logits_tensor,
                            resized_image_tensor,
                            session,
                            image_bytes_feed,
                            endpoints,
                            batch_size):
    """
    """
    logits, batch_images, batch_endpoints = session.run(
        fetches=[logits_tensor, resized_image_tensor, endpoints],
        feed_dict={image_bytes_feed: frames})

    batch_heatmaps = []
    for image_index in range(batch_size):
        _, threshold = cv2.threshold(logits[image_index, ...],
                                     -0.5,
                                     255,
                                     cv2.THRESH_BINARY)

        heatmap_on_image = np.zeros(batch_images[image_index].shape, dtype=np.uint8)
        for joint_index in range(logits.shape[-1]):
            if joint_index in [0, 1, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]:
                dist_transform = cv2.distanceTransform(threshold[..., joint_index].astype(np.uint8),
                                                       cv2.DIST_L1,
                                                       3)
                dist_transform = cv2.convertScaleAbs(dist_transform,
                                                     dist_transform,
                                                     255.0/np.max(dist_transform))
                heatmap = cv2.applyColorMap(dist_transform, cv2.COLORMAP_JET)
                heatmap[..., 0] = 0

                heatmap_on_image += heatmap

        heatmap_on_image = np.clip(heatmap_on_image, 0, 255)
        # alpha = 0.5
        # heatmap_on_image = cv2.addWeighted(batch_images[image_index],
        #                                    alpha,
        #                                    heatmap_on_image,
        #                                    1.0 - alpha,
        #                                    0.0)

        batch_heatmaps.append(heatmap_on_image)

    return batch_heatmaps, batch_endpoints


def TFHttpRequestHandlerFactory(session,
                                image_bytes_feed,
                                logits_tensor,
                                resized_image_tensor,
                                endpoints):
    """This function returns subclasses of
    `http.server.BaseHTTPRequestHandler`, using the closure of the function
    call to allow extra parameters (namely the session to run a computation
    graph for which `logits_tensor` is the output, and the placeholder to feed
    that graph) to be "local" to the class.

    This is necessary because the constructor of the subclass of
    `BaseHTTPRequestHandler` expects a certain function signature.
    """
    class TFHttpRequestHandler(http.server.BaseHTTPRequestHandler):
        """Defines handlers for specific HTTP request codes."""
        def __init__(self, *args, **kwargs):
            super(TFHttpRequestHandler, self).__init__(*args, **kwargs)

        def _send_response_headers(self, json_string):
            """
            """
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            # self.send_header('Access-Control-Allow-Origin', 'https://brendanduke.ca')
            self.send_header('Access-Control-Allow-Origin', 'http://localhost:5000')
            self.end_headers()

            self.wfile.write(json_string.encode('utf8'))

        def _respond_with_joints(self, frames):
            """Takes `frames` and does joint inference on it, returning a 200 OK
            HTTP response with the response data set being a JSON string
            containing inferred joint positions.
            """
            joint_predictions_json = _get_image_joint_predictions(
                frames,
                session,
                image_bytes_feed,
                logits_tensor)
            self._send_response_headers(joint_predictions_json)

        @timethis
        def _respond_with_heatmaps(self, frames):
            """
            """
            batch_heatmaps, batch_endpoints = _get_heatmaps_for_batch(
                frames,
                logits_tensor,
                resized_image_tensor,
                session,
                image_bytes_feed,
                endpoints,
                BATCH_SIZE)

            # heatmap_jpegs = tf.map_fn(
            #     fn=lambda image: tf.image.encode_jpeg(image=image),
            #     elems=batch_heatmaps,
            #     parallel_iterations=len(batch_heatmaps))

            # heatmap_jpegs = session.run(heatmap_jpegs)
            heatmap_jpegs = batch_heatmaps

            b64_heatmap_jpegs = []
            for heatmap in heatmap_jpegs:
                # b64_heatmap_jpegs.append(base64.b64encode(heatmap_jpegs))
                with tf.Graph().as_default():
                    with tf.Session() as encode_sess:
                        heatmap_jpeg = encode_sess.run(tf.image.encode_jpeg(image=heatmap))
                b64_heatmap_jpeg = base64.b64encode(heatmap_jpeg).decode('utf-8')
                b64_heatmap_jpegs.append(b64_heatmap_jpeg)

            self._send_response_headers(json.dumps(b64_heatmap_jpegs))

        def do_GET(self):
            """This implementation of an HTTP GET request handler will take an
            image URL (JPEG) passed as an option parameter
            (image_url=<jpeg-url>) in the request URL, do joint inference on
            the image and return a JSON string of the estimated joints.

            This function is mainly for debugging, e.g. that the server is up.
            E.g., the below curl command should return a string of JSON.

            curl -X GET https://brendanduke.ca:8765/?image_url=http://st2.depositphotos.com/1912333/10089/i/950/depositphotos_100892946-stock-photo-sporty-woman-waving-hands.jpg --insecure
            """
            image_url = self.requestline.split()[1]
            image_url = urllib.parse.unquote(image_url)
            image_url = re.match('/\?image_url=(.*)', image_url)
            if image_url is None:
                self.send_response(400)
                self.send_header('Access-Control-Allow-Origin', 'https://brendanduke.ca')
                self.end_headers()
                return

            image_url = image_url.groups()[0]

            image_request = requests.get(image_url)

            self._respond_with_joints(image_request.content)

        @timethis
        def do_POST(self):
            """HTTP POST requests should contain a JPEG image encoded in base
            64 as the request data.

            The server will attempt to decode the base 64 image and do
            joint-position inference, returning a JSON string representing the
            inferred joint positions.
            """
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')

            jpeg_images = json.loads(post_data)
            frames = []
            for image in jpeg_images:
                decoded_img = base64.b64decode(image)
                frames.append(decoded_img)

            post_url = self.requestline.split()[1]
            if re.match('/heatmap', post_url) is not None:
                self._respond_with_heatmaps(frames)
            else:
                self._respond_with_joints(frames)

    return TFHttpRequestHandler


def _get_joint_position_inference_graph(image_bytes_feed):
    """This function sets up a computation graph that will decode from JPEG the
    input placeholder image `image_bytes_feed`, pad and resize it to shape
    [IMAGE_DIM, IMAGE_DIM], then run human pose inference on the image using
    the "Two VGG-16s cascade" model.
    """
    decoded_image = tf.map_fn(
        fn=lambda image: tf.image.decode_jpeg(contents=image),
        elems=image_bytes_feed,
        dtype=tf.uint8,
        parallel_iterations=BATCH_SIZE)

    decoded_image = tf.image.convert_image_dtype(image=decoded_image,
                                                 dtype=tf.float32)

    normalized_image = tf.subtract(x=decoded_image, y=0.5)
    normalized_image = tf.multiply(x=normalized_image, y=2.0)

    normalized_image = tf.reshape(tensor=normalized_image,
                                  shape=[BATCH_SIZE, IMAGE_DIM, IMAGE_DIM, 3])

    with tf.device(device_name_or_function='/gpu:0'):
        with slim.arg_scope([slim.model_variable], device='/cpu:0'):
            with slim.arg_scope(resnet_bulat.resnet_arg_scope()):
                logits, endpoints = resnet_bulat.resnet_50_detector(normalized_image,
                                                                    16,
                                                                    False,
                                                                    False)

    return (logits,
            tf.image.convert_image_dtype(image=decoded_image, dtype=tf.uint8),
            endpoints)


def run():
    """Starts a server that will handle  HTTP requests to use TensorFlow.

    At server startup, a joint inference computation graph is setup, a session
    to run the graph in is created, and model weights are restored from
    `RESTORE_PATH`.

    The server listens on an SSL-wrapped socket at port 8765 of localhost,
    using an SSL certificate obtained from https://letsencrypt.org/.
    """
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            image_bytes_feed = tf.placeholder(dtype=tf.string)

            logits, resized_image, endpoints = _get_joint_position_inference_graph(
                image_bytes_feed)

            session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=RESTORE_PATH)
            assert latest_checkpoint is not None

            variables_to_restore = {
                    var.op.name.replace('resnet_v1_50_pyramid', 'pyramid'): var for var in tf.global_variables()
            }
            restorer = tf.train.Saver(var_list=variables_to_restore)
            restorer.restore(sess=session, save_path=latest_checkpoint)

            request_handler = TFHttpRequestHandlerFactory(session,
                                                          image_bytes_feed,
                                                          logits,
                                                          resized_image,
                                                          endpoints)
            server_address = ('localhost', 8765)
            httpd = http.server.HTTPServer(server_address, request_handler)
            httpd.socket = ssl.wrap_socket(httpd.socket,
                                           keyfile='./domain.key',
                                           certfile='./signed.crt',
                                           server_side=True)

            print('Serving!')
            httpd.serve_forever()


if __name__ == "__main__":
    run()
