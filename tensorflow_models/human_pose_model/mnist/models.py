import tensorflow as tf
import tensorflow.contrib.slim as slim

class LogisticRegression(object):

    def __init__(self,input_imgs):
        # our input
        self.input_imgs = input_imgs

    def network(self):
        # Collect outputs for fully connected
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                           outputs_collections='fully_connected_collection'):

            logits = slim.fully_connected(self.input_imgs, 10, scope='fc1')
            end_points = slim.utils.convert_collection_to_dict('fully_connected_collection')

        return logits, end_points

