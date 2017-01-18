import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg
from mpii_read import Person
from sparse_to_dense import sparse_joints_to_dense
from input_pipeline import setup_eval_input_pipeline

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '.',
                           """Path to take input TFRecord files from.""")
tf.app.flags.DEFINE_string('restore_path', None,
                           """Path to take checkpoint file from.""")
tf.app.flags.DEFINE_integer('image_dim', 299,
                            """Dimension of the square input image.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of threads to use to preprocess
                            images.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Size of each mini-batch (number of examples
                            processed at once).""")

JOINT_NAMES = ['0 - r ankle',
               '1 - r knee',
               '2 - r hip',
               '3 - l hip',
               '4 - l knee',
               '5 - l ankle',
               '6 - pelvis',
               '7 - thorax',
               '8 - upper neck',
               '9 - head top',
               '10 - r wrist',
               '11 - r elbow',
               '12 - r shoulder',
               '13 - l shoulder',
               '14 - l elbow',
               '15 - l wrist']

def _get_points_from_flattened_joints(x_joints, y_joints, batch_size):
    """Takes in a list of batches of x and y joint coordinates, and returns a
    list of batches of tuples (x, y) of joint points.
    """
    joint_points = []
    for batch_index in range(batch_size):
        joint_points.append([(x, y) for (x, y) in zip(x_joints[batch_index], y_joints[batch_index])])

    return np.array(joint_points)


def evaluate():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            eval_batch = setup_eval_input_pipeline(FLAGS.data_dir,
                                                   FLAGS.batch_size,
                                                   FLAGS.num_preprocess_threads,
                                                   FLAGS.image_dim)

            with tf.device(device_name_or_function='/gpu:0'):
                with slim.arg_scope([slim.model_variable], device='/cpu:0'):
                    with slim.arg_scope(vgg.vgg_arg_scope()):
                        logits, _ = vgg.vgg_16(inputs=eval_batch.images,
                                               num_classes=2*Person.NUM_JOINTS)

            next_x_gt_joints, next_y_gt_joints, next_weights = sparse_joints_to_dense(
                eval_batch, Person.NUM_JOINTS)

            restorer = tf.train.Saver()

            session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            restorer.restore(sess=session, save_path=FLAGS.restore_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            sum_squared_loss = 0
            num_test_data = 1024
            num_batches = int(num_test_data/FLAGS.batch_size)
            matched_joints = Person.NUM_JOINTS*[0]
            predicted_joints = Person.NUM_JOINTS*[0]
            for _ in range(num_batches):
                [predictions, x_gt_joints, y_gt_joints, weights, head_size] = session.run(
                    fetches=[logits, next_x_gt_joints, next_y_gt_joints, next_weights, eval_batch.head_size])

                gt_joint_points = _get_points_from_flattened_joints(
                    x_gt_joints, y_gt_joints, FLAGS.batch_size)

                predicted_joint_points = _get_points_from_flattened_joints(
                    predictions[:, 0:Person.NUM_JOINTS],
                    predictions[:, Person.NUM_JOINTS:],
                    FLAGS.batch_size)

                # NOTE(brendan): Here we are following steps to calculate the
                # PCKh metric, which defins a joint estimate as matching the
                # ground truth if the estimate lies within 50% of the head
                # segment length. Head segment length is defined as the
                # diagonal across the annotated head rectangle in the MPII
                # data, multiplied by a factor of 0.6.
                joint_weights = weights[:, 0:Person.NUM_JOINTS]

                distance_from_gt = np.sqrt(np.sum(np.square(gt_joint_points - predicted_joint_points), 2))
                matched_joints += np.sum(joint_weights*(distance_from_gt < 0.5*head_size), 0)

                predicted_joints += np.sum(joint_weights, 0)

                gt_joints = np.concatenate((x_gt_joints, y_gt_joints), 1)
                sum_squared_loss += np.sum(weights*np.square(predictions - gt_joints))

            print('Matched joints:', matched_joints)
            print('Predicted joints:', predicted_joints)

            PCKh = matched_joints/predicted_joints
            print('PCKh:')
            for joint_index in range(Person.NUM_JOINTS):
                print(JOINT_NAMES[joint_index], ':', PCKh[joint_index])

            print('Average squared loss:', sum_squared_loss/num_test_data)

            coord.request_stop()
            coord.join(threads)


def main(argv=None):
    evaluate()


if __name__ == "__main__":
    tf.app.run()
