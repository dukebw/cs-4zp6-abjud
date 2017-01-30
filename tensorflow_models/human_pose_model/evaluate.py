import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import NETS, NET_ARG_SCOPES
from mpii_read import Person
from sparse_to_dense import sparse_joints_to_dense
from input_pipeline import setup_eval_input_pipeline

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


def evaluate(network_name,
             data_dir,
             restore_path,
             log_file,
             image_dim,
             num_preprocess_threads,
             batch_size,
             epoch=0):
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            eval_batch = setup_eval_input_pipeline(data_dir,
                                                   batch_size,
                                                   num_preprocess_threads,
                                                   image_dim)

            part_detect_net = NETS[network_name]
            net_arg_scope = NET_ARG_SCOPES[network_name]

            with tf.device(device_name_or_function='/gpu:0'):
                with slim.arg_scope([slim.model_variable], device='/cpu:0'):
                    with slim.arg_scope(net_arg_scope()):
                        logits, _ = part_detect_net(inputs=eval_batch.images,
                                                    num_classes=Person.NUM_JOINTS)

            next_x_gt_joints, next_y_gt_joints, next_weights = sparse_joints_to_dense(
                eval_batch, Person.NUM_JOINTS)

            restorer = tf.train.Saver()

            session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            restorer.restore(sess=session, save_path=restore_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            num_test_data = 1024
            num_batches = int(num_test_data/batch_size)
            matched_joints = Person.NUM_JOINTS*[0]
            predicted_joints = Person.NUM_JOINTS*[0]
            for _ in range(num_batches):
                [predictions, x_gt_joints, y_gt_joints, weights, head_size] = session.run(
                    fetches=[logits, next_x_gt_joints, next_y_gt_joints, next_weights, eval_batch.head_size])

                head_size = np.reshape(head_size, [batch_size, 1])

                gt_joint_points = _get_points_from_flattened_joints(
                    x_gt_joints, y_gt_joints, batch_size)

                x_predicted_joints = np.empty((batch_size, Person.NUM_JOINTS))
                y_predicted_joints = np.empty((batch_size, Person.NUM_JOINTS))
                for batch_index in range(batch_size):
                    for joint_index in range(Person.NUM_JOINTS):
                        joint_heatmap = predictions[batch_index, ..., joint_index]
                        xy_max_confidence = np.unravel_index(
                            joint_heatmap.argmax(), joint_heatmap.shape)
                        y_predicted_joints[batch_index, joint_index] = xy_max_confidence[0]/image_dim - 0.5
                        x_predicted_joints[batch_index, joint_index] = xy_max_confidence[1]/image_dim - 0.5

                predicted_joint_points = _get_points_from_flattened_joints(
                    x_predicted_joints,
                    y_predicted_joints,
                    batch_size)

                # NOTE(brendan): Here we are following steps to calculate the
                # PCKh metric, which defines a joint estimate as matching the
                # ground truth if the estimate lies within 50% of the head
                # segment length. Head segment length is defined as the
                # diagonal across the annotated head rectangle in the MPII
                # data, multiplied by a factor of 0.6.
                joint_weights = weights[:, 0:Person.NUM_JOINTS]

                distance_from_gt = np.sqrt(np.sum(np.square(gt_joint_points - predicted_joint_points), 2))
                matched_joints += np.sum(joint_weights*(distance_from_gt < 0.5*head_size), 0)

                predicted_joints += np.sum(joint_weights, 0)

                gt_joints = np.concatenate((x_gt_joints, y_gt_joints), 1)

            log_file_handle = open(log_file, 'a')

            if (epoch > 0):
                log_file_handle.write('\n')
            log_file_handle.write('************************************************\n')
            log_file_handle.write('Epoch {} PCKh metric.\n'.format(epoch))
            log_file_handle.write('************************************************\n\n')
            log_file_handle.write('Matched joints: {}\n'.format(matched_joints))
            log_file_handle.write('Predicted joints: {}\n'.format(predicted_joints))

            PCKh = matched_joints/predicted_joints
            log_file_handle.write('PCKh:\n')
            for joint_index in range(Person.NUM_JOINTS):
                log_file_handle.write('{}: {}\n'.format(JOINT_NAMES[joint_index], PCKh[joint_index]))
            log_file_handle.write('\nTotal PCKh: {}\n'.format(np.sum(PCKh)/len(PCKh)))

            log_file_handle.close()

            coord.request_stop()
            coord.join(threads)


def main(argv=None):
    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string('network_name', None,
                               """Name of desired network to use for part
                               detection. Valid options: vgg, inception_v3.""")
    tf.app.flags.DEFINE_string('data_dir', '.',
                               """Path to take input TFRecord files from.""")
    tf.app.flags.DEFINE_string('restore_path', None,
                               """Path to take checkpoint file from.""")
    tf.app.flags.DEFINE_string('log_file', './log/eval_log',
                               """Relative filepath to log evaluation metrics
                               to.""")
    tf.app.flags.DEFINE_integer('image_dim', 299,
                                """Dimension of the square input image.""")
    tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                                """Number of threads to use to preprocess
                                images.""")
    tf.app.flags.DEFINE_integer('batch_size', 32,
                                """Size of each mini-batch (number of examples
                                processed at once).""")

    evaluate(FLAGS.network_name,
             FLAGS.data_dir,
             FLAGS.restore_path,
             FLAGS.log_file,
             FLAGS.image_dim,
             FLAGS.num_preprocess_threads,
             FLAGS.batch_size)


if __name__ == "__main__":
    tf.app.run()
