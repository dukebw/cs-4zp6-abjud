"""This module implements the TensorFlow HTTP server, which is a standalone
server that is able to handle HTTP requests to do TensorFlow operations.
"""
import re
import urllib
import http.server
import requests
import tensorflow as tf
from human_pose_model.networks.vgg_bulat import two_vgg_16s_cascade

# @debug
from human_pose_model.pose_utils.img_utils import draw_logits

RESTORE_PATH = '/mnt/data/datasets/MPII_HumanPose/logs/vgg_bulat/both_nets_xentropy_regression/23'
IMAGE_DIM = 380

def TFHttpRequestHandlerFactory(session, image_bytes_feed, logits_tensor, resized_image):
    class TFHttpRequestHandler(http.server.BaseHTTPRequestHandler):
        """Defines handlers for specific HTTP request codes."""
        def __init__(self, *args, **kwargs):
            super(TFHttpRequestHandler, self).__init__(*args, **kwargs)

        def do_GET(self):
            image_url = self.requestline.split()[1]
            image_url = urllib.parse.unquote(image_url)
            image_url = re.match('/\?image_url=(.*)', image_url)
            assert image_url is not None

            image_url = image_url.groups()[0]

            image_request = requests.get(image_url)

            feed_dict = {
                image_bytes_feed: image_request.content
            }
            logits = session.run(fetches=logits_tensor, feed_dict=feed_dict)

            # @debug
            image = session.run(tf.image.convert_image_dtype(image=resized_image, dtype=tf.uint8), feed_dict=feed_dict)
            draw_logits(image, logits)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write('{"response": "Hello!"}'.encode('utf8'))

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
                                                      logits,
                                                      resized_image)
        server_address = ('localhost', 8765)
        httpd = http.server.HTTPServer(server_address, request_handler)
        httpd.serve_forever()


if __name__ == "__main__":
    run()
