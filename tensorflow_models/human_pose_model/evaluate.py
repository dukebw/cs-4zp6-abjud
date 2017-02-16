import math
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.contrib.slim as slim
from mpii_read import Person
from sparse_to_dense import sparse_joints_to_dense
from input_pipeline import setup_eval_input_pipeline
from nets import inference

NUM_JOINTS = 16
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
             loss_name,
             data_dir,
             restore_path,
             log_file,
             image_dim,
             num_preprocess_threads,
             batch_size,
             heatmap_stddev_pixels,
             epoch=0):
    """
    """
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            num_gpus = 4
            data_filenames = tf.gfile.Glob(
                os.path.join(data_dir, 'valid*tfrecord'))
            assert data_filenames, ('No data files found.')

            min_examples_per_shard = math.inf
            options = tf.python_io.TFRecordOptions(
                compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
            for data_file in data_filenames:
                examples_in_shard = 0
                for _ in tf.python_io.tf_record_iterator(path=data_file, options=options):
                    examples_in_shard += 1

                min_examples_per_shard = min(min_examples_per_shard, examples_in_shard)

            eval_batch = setup_eval_input_pipeline(batch_size,
                                                   num_preprocess_threads,
                                                   image_dim,
                                                   heatmap_stddev_pixels,
                                                   data_filenames)
            # sets up the evaluation op
            images_split = tf.split(axis=0, num_or_size_splits=num_gpus, value=eval_batch.images)
            binary_maps_split = tf.split(axis=0, num_or_size_splits=num_gpus, value=eval_batch.binary_maps)
            heatmaps_split = tf.split(axis=0, num_or_size_splits=num_gpus, value=eval_batch.heatmaps)
            weights_split = tf.split(axis=0, num_or_size_splits=num_gpus, value=eval_batch.weights)

            tower_logits_list = []
            total_loss = tf.constant(0,dtype=tf.float32)
            for gpu_index in range(num_gpus):
                with tf.device(device_name_or_function='/gpu:{}'.format(gpu_index)):
                    with tf.name_scope('tower_{}'.format(gpu_index)) as scope:
                        loss, logits = inference(images_split[gpu_index],
                                                 binary_maps_split[gpu_index],
                                                 heatmaps_split[gpu_index],
                                                 weights_split[gpu_index],
                                                 gpu_index,
                                                 network_name,
                                                 loss_name,
                                                 scope)

                        tower_logits_list.append(logits)
                        total_loss += loss

            total_loss /= num_gpus
            tf.summary.scalar(name='total_loss', tensor=total_loss)

            # Concatenate the outputs of the gpus along batch dim
            tower_logits = tf.concat(axis=0, values=tower_logits_list)
            merged_logits = tf.reshape(tf.reduce_max(logits, 3),
                                [int(images_split[0].get_shape()[0]), image_dim, image_dim, 1])
            merged_logits = tf.cast(merged_logits, tf.float32)
            tf.summary.image(name='logits', tensor=merged_logits)
            next_x_gt_joints, next_y_gt_joints, next_weights = sparse_joints_to_dense(
                eval_batch, Person.NUM_JOINTS)

            restorer = tf.train.Saver()

            session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            restorer.restore(sess=session, save_path=restore_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            num_test_examples = len(data_filenames)*min_examples_per_shard
            num_batches = int(num_test_examples/batch_size)
            matched_joints = Person.NUM_JOINTS*[0]
            predicted_joints = Person.NUM_JOINTS*[0]
            valid_epoch_mean_loss = 0
            for _ in tqdm(range(num_batches)):
                [batch_loss, predictions, x_gt_joints, y_gt_joints, weights, head_size] = session.run(
                    fetches=[loss, tower_logits, next_x_gt_joints, next_y_gt_joints, next_weights, eval_batch.head_size])

                valid_epoch_mean_loss += batch_loss
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

            if (epoch > 1):
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
            valid_epoch_mean_loss /= num_batches
            log_file_handle.write('\nValidation Loss: {}\n'.format(valid_epoch_mean_loss))

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
    tf.app.flags.DEFINE_string('log_filepath', './log/temp/eval_log',
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
             FLAGS.loss_name,
             FLAGS.data_dir,
             FLAGS.restore_path,
             FLAGS.log_filepath,
             FLAGS.image_dim,
             FLAGS.num_preprocess_threads,
             FLAGS.heatmap_stddev_pixels,
             FLAGS.batch_size)


if __name__ == "__main__":
    tf.app.run()
