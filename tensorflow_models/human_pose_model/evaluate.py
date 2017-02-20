import math
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pose_util
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

def setup_val_loss_op(num_gpus, eval_batch, image_dim, network_name, loss_name):
    """Creates the inference part of the validation graph, and returns the
    total loss calculated across all `num_gpus` used to do the evaluation.

    Also returns the concatenated list of per-tower logits, which are used as
    predictions for the batch of examples.
    """
    images_split = tf.split(axis=0,
                            num_or_size_splits=num_gpus,
                            value=eval_batch.images)
    binary_maps_split = tf.split(axis=0,
                                 num_or_size_splits=num_gpus,
                                 value=eval_batch.binary_maps)
    heatmaps_split = tf.split(axis=0,
                              num_or_size_splits=num_gpus,
                              value=eval_batch.heatmaps)
    weights_split = tf.split(axis=0,
                             num_or_size_splits=num_gpus,
                             value=eval_batch.weights)

    tower_logits_list = []
    total_loss = tf.constant(0, dtype=tf.float32)
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
                                         False,
                                         scope)

                tower_logits_list.append(logits)
                total_loss += loss

    total_loss /= num_gpus

    tower_logits = tf.concat(axis=0, values=tower_logits_list)

    tf.summary.scalar(name='total_loss', tensor=total_loss)

    per_gpu_batch_size = int(images_split[0].get_shape()[0])
    merged_logits = tf.reshape(
        tf.reduce_max(logits, 3),
        [per_gpu_batch_size, image_dim, image_dim, 1])
    merged_logits = tf.cast(merged_logits, tf.float32)
    tf.summary.image(name='logits', tensor=merged_logits)

    return total_loss, tower_logits


def setup_evaluation(FLAGS):
    """Sets up the entire input pipeline and inference graph for evaluation,
    using interfaces exposed from `evaluate.py`.

    Returns:
        (val_fetches, val_session) tuple, where val_fetches must be kept
        synchronized with the outputs from the session.run() call in
        evaluate.py.
    """
    num_counting_threads = FLAGS.num_preprocess_threads + FLAGS.num_readers
    num_val_examples, val_data_filenames = pose_util.count_training_examples(
        FLAGS.validation_data_dir, num_counting_threads, 'valid')

    eval_batch = setup_eval_input_pipeline(FLAGS.batch_size,
                                           FLAGS.num_preprocess_threads,
                                           FLAGS.image_dim,
                                           FLAGS.heatmap_stddev_pixels,
                                           val_data_filenames)

    val_loss, val_tower_logits = setup_val_loss_op(FLAGS.num_gpus,
                                                   eval_batch,
                                                   FLAGS.image_dim,
                                                   FLAGS.network_name,
                                                   FLAGS.loss_name)

    next_x_gt_joints, next_y_gt_joints, next_weights = sparse_joints_to_dense(
        eval_batch, Person.NUM_JOINTS)

    gt_data = [next_x_gt_joints, next_y_gt_joints, next_weights, eval_batch.head_size]

    return num_val_examples, val_loss, val_tower_logits, gt_data


def evaluate_single_epoch(restorer,
                          restore_path,
                          loss,
                          logits,
                          gt_data,
                          num_val_examples,
                          batch_size,
                          image_dim,
                          epoch,
                          log_file_handle):
    """Evaluates the model checkpoint given by `restore_path` using the PCKh
    metric.

    The entire graph is assumed to have been constructed in `session`, such
    that all there is to run validation is to run `val_fetches`, and compute
    relevant metrics.
    """
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        with tf.device('/cpu:0'):
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=restore_path)
            assert latest_checkpoint is not None

            restorer.restore(sess=session, save_path=latest_checkpoint)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            num_batches = int(math.ceil(num_val_examples/batch_size))
            matched_joints = Person.NUM_JOINTS*[0]
            predicted_joints = Person.NUM_JOINTS*[0]
            valid_epoch_mean_loss = 0
            for _ in tqdm(range(num_batches)):
                [batch_loss, predictions, x_gt_joints, y_gt_joints, weights, head_size] = session.run(
                    fetches=[loss, logits] + gt_data)

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

            if (epoch > 1):
                log_file_handle.write('\n')
            valid_epoch_mean_loss /= num_batches
            log_file_handle.write('\nMean validation loss: {}\n\n'.format(valid_epoch_mean_loss))
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
            log_file_handle.flush()

            coord.request_stop()
            coord.join(threads=threads)


def evaluate():
    """Does a loop checking for new checkpoints and evaluating them all, then
    exits.

    TODO(brendan):

    1. Check for `last_evaluated_checkpoint.json` in `FLAGS.log_dir`.
    2. When the process starts, checks the last evaluated checkpoint, and
       evaluates all checkpoints present that came later.
    3. Once there are no new checkpoints, the evaluation process exits.
    """
    num_val_examples, val_loss, val_logits, gt_data = setup_evaluation(FLAGS)

    restorer = tf.train.Saver(var_list=tf.global_variables())

    evaluate_single_epoch(restorer,
                          restore_path,
                          val_loss,
                          val_logits,
                          gt_data,
                          num_val_examples,
                          FLAGS.batch_size,
                          FLAGS.image_dim,
                          epoch,
                          log_file_handle)

def main(argv=None):
    """Usage: python3 -m evaluate
    """
    evaluate()

if __name__ == "__main__":
    tf.app.run()
