import os
import time
import numpy as np
from tqdm import trange
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from pose_utils import pose_util
from pose_utils.pose_flags import FLAGS
from dataset.mpii_datatypes import Person
from input_pipeline import setup_train_input_pipeline
from networks.inference import inference
from evaluate import setup_evaluation, evaluate_single_epoch
import pdb
import pandas as pd

RMSPROP_DECAY = 0.9
RMSPROP_MOMENTUM = 0.9
RMSPROP_EPSILON = 1.0


def _setup_adam_optimizer():
    global_step = tf.get_variable(
        name='global_step',
        shape=[],
        dtype=tf.int64,
        initializer=tf.constant_initializer(0),
        trainable=False)

    optimizer = tf.train.AdamOptimizer()

    return global_step, optimizer

# Legacy?
def _setup_optimizer(batches_per_epoch,
                     num_epochs_per_decay,
                     initial_learning_rate,
                     learning_rate_decay_factor):
    """Sets up the optimizer
    [RMSProp]((http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf))
    and learning rate decay schedule.

    Args:
        batches_per_epoch: Number of batches in an epoch.
        num_epochs_per_decay: Number of full runs through of the data set per
            learning rate decay.
        initial_learning_rate: Learning rate to start with on step 0.
        learning_rate_decay_factor: Factor by which to decay the learning rate.

    Returns:
        global_step: Counter to be incremented each training step, used to
            calculate the decaying learning rate.
        optimizer: `RMSPropOptimizer`, minimizes loss function by gradient
            descent (RMSProp).
    """
    global_step = tf.get_variable(
        name='global_step',
        shape=[],
        dtype=tf.int64,
        initializer=tf.constant_initializer(0),
        trainable=False)

    decay_steps = batches_per_epoch*num_epochs_per_decay
    learning_rate = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=decay_steps,
                                               decay_rate=learning_rate_decay_factor)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          decay=RMSPROP_DECAY,
                                          momentum=RMSPROP_MOMENTUM,
                                          epsilon=RMSPROP_EPSILON)

    tf.summary.scalar(name='learning_rate', tensor=learning_rate)

    return global_step, optimizer


def _get_variables_to_train():
    """Returns the set of trainable variables, given by the `trainable_scopes`
    flag if passed, or all trainable variables otherwise.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()

    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=scope)
        variables_to_train.extend(variables)

    return variables_to_train


def _average_gradients(tower_grads):
    """Averages the gradients for all gradients in `tower_grads`, and returns a
    list of `(avg_grad, var)` tuples.
    """
    avg_grad_and_vars = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for grad, _ in grad_and_vars:
            grads.append(tf.expand_dims(input=grad, axis=0))

        grad = tf.concat(values=grads, axis=0)
        grad = tf.reduce_mean(input_tensor=grad, axis=0)

        avg_grad_and_vars.append((grad, grad_and_vars[0][1]))

    return avg_grad_and_vars


def _setup_training_op(images,
                       binary_maps,
                       heatmaps,
                       weights,
                       is_visible_weights,
                       global_step,
                       optimizer,
                       num_gpus):
    """Sets up inference (predictions), loss calculation, and minimization
    based on the input optimizer.

    Args:
        images: Batch of preprocessed examples dequeued from the input
            pipeline.
        heatmaps: Confidence maps of ground truth joints.
        weights: Weights of heatmaps (tensors of all 1s if joint present, all
            0s if not present).
        global_step: Training step counter.
        optimizer: Optimizer to minimize the loss function.
        num_gpus: Number of GPUs to split each mini-batch across.

    Returns: Operation to run a training step.
    """
    images_split = tf.split(value=images, num_or_size_splits=num_gpus, axis=0)
    binary_maps_split = tf.split(value=binary_maps,
                                 num_or_size_splits=num_gpus,
                                 axis=0)
    heatmaps_split = tf.split(value=heatmaps,
                              num_or_size_splits=num_gpus,
                              axis=0)
    weights_split = tf.split(value=weights,
                             num_or_size_splits=num_gpus,
                             axis=0)
    is_visible_weights_split = tf.split(value=is_visible_weights,
                                        num_or_size_splits=num_gpus,
                                        axis=0)

    tower_grads = []
    for gpu_index in range(num_gpus):
        with tf.device(device_name_or_function='/gpu:{}'.format(gpu_index)):
            with tf.name_scope('tower_{}'.format(gpu_index)) as scope:
                # TODO (Thor) generate new samples by extrapolation or interpolation
                #
                #
                total_loss, _ = inference(images_split[gpu_index],
                                          binary_maps_split[gpu_index],
                                          heatmaps_split[gpu_index],
                                          weights_split[gpu_index],
                                          is_visible_weights_split[gpu_index],
                                          gpu_index,
                                          FLAGS.network_name,
                                          FLAGS.loss_name,
                                          FLAGS.is_detector_training,
                                          True,
                                          scope)

                batchnorm_updates = tf.get_collection(
                    key=tf.GraphKeys.UPDATE_OPS, scope=scope)

                with ops.control_dependencies(batchnorm_updates):
                    barrier = control_flow_ops.no_op(name='update_barrier')
                total_loss = control_flow_ops.with_dependencies(
                    [barrier], total_loss)

                grads = optimizer.compute_gradients(
                    loss=total_loss, var_list=_get_variables_to_train())
                tower_grads.append(grads)

    avg_grad_and_vars = _average_gradients(tower_grads)

    apply_gradient_op = optimizer.apply_gradients(
        grads_and_vars=avg_grad_and_vars,
        global_step=global_step)

    # TODO(brendan): It is possible to keep track of moving averages of
    # variables ("shadow variables"), and these shadow variables can be used
    # for evaluation.
    #
    # See TF models ImageNet Inception training code.

    return apply_gradient_op, total_loss


def _restore_checkpoint_variables(session,
                                  global_step,
                                  checkpoint_path,
                                  checkpoint_exclude_scopes,
                                  is_regression_subnetwork_pretrained):
    """Initializes the model in the graph of a passed session with the
    variables in the file found in `checkpoint_path`, except those excluded by
    `checkpoint_exclude_scopes`.
    """
    if checkpoint_path is None:
        return

    if checkpoint_exclude_scopes is None:
        variables_to_restore = tf.global_variables()
    else:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]

        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if exclusion in var.op.name:
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

    if FLAGS.restore_global_step:
        variables_to_restore.append(global_step)

    if is_regression_subnetwork_pretrained:
        variables_to_restore = {var.op.name.replace('vgg_16_regression', 'vgg_16'): var for var in variables_to_restore}

    restorer = tf.train.Saver(var_list=variables_to_restore)
    restorer.restore(sess=session, save_path=checkpoint_path)


def _train_single_epoch(session,
                        saver,
                        train_writer,
                        train_op,
                        loss,
                        global_step,
                        summary_op,
                        num_batches_per_epoch,
                        log_handle,
                        log_dir,
                        runtime_df):
    """Runs the training loop, executing the graph stored in `session`,
    and for every epoch executes the graph in `val_session`, which will
    evaluate the latest model checkpoint on a validation set.

    After the epoch, a checkpoint is saved to `log_dir`.

    Returns:
        epoch: The epoch number that was just trained, calculated from the
               `global_step` variable.
    """
    runtime_df['Epoch'].add(0)
    train_epoch_mean_loss = 0
    Epoch = trange(num_batches_per_epoch, desc='Loss', leave=True)
    for batch_step in Epoch:
        start_time = time.time()
        _, batch_loss, total_steps = session.run(fetches=[train_op,
                                                          loss,
                                                          global_step])

        duration = time.time() - start_time

        step_desc = ('step {}: loss = {} ({:.2f} sec/step)'
                     .format(total_steps, batch_loss, duration))
        train_epoch_mean_loss += batch_loss
        Epoch.set_description(step_desc)
        Epoch.refresh()

        assert not np.isnan(batch_loss)

        if (total_steps % 100) == 0:
            summary_str = session.run(summary_op)
            train_writer.add_summary(summary=summary_str,
                                     global_step=total_steps)

    checkpoint_path = os.path.join(log_dir, FLAGS.checkpoint_name+'.ckpt')
    saver.save(sess=session,
               save_path=checkpoint_path,
               global_step=total_steps)

    epoch = int(total_steps/num_batches_per_epoch)
    train_epoch_mean_loss /= num_batches_per_epoch
    log_handle.write('Epoch {} done.\n'.format(epoch))
    log_handle.write('Mean training loss is {}.\n'.format(train_epoch_mean_loss))
    log_handle.flush()
    runtime_df['Mean_Training_Loss'].add(train_epoch_mean_loss)
    return runtime_df


def _setup_training(FLAGS):
    """Sets up the entire training graph, including input pipeline, inference
    back-propagation and summaries.

    Args:
        FLAGS: The set of flags passed via command line. See the definitions at
            the top of this file.

    Returns:
        (num_batches_per_epoch, train_op, train_loss, global_step) tuple needed
        to run training steps.
    """
    num_counting_threads = FLAGS.num_preprocess_threads + FLAGS.num_readers
    num_training_examples, train_data_filenames = pose_util.count_training_examples(
        FLAGS.train_data_dir, num_counting_threads, 'train')

    images, binary_maps, heatmaps, weights, is_visible_weights = setup_train_input_pipeline(FLAGS, train_data_filenames)

    num_batches_per_epoch = int(num_training_examples / FLAGS.batch_size)

    if FLAGS.optimizer == 'adam':
        global_step, optimizer = _setup_adam_optimizer()
    else:
        global_step, optimizer = _setup_optimizer(num_batches_per_epoch,
                                                  FLAGS.num_epochs_per_decay,
                                                  FLAGS.initial_learning_rate,
                                                  FLAGS.learning_rate_decay_factor)

    train_op, train_loss = _setup_training_op(images,
                                              binary_maps,
                                              heatmaps,
                                              weights,
                                              is_visible_weights,
                                              global_step,
                                              optimizer,
                                              FLAGS.num_gpus)

    return num_batches_per_epoch, train_op, train_loss, global_step


def _init_regression_subnetwork_first_layer(session, second_checkpoint_path):
    """Initializes a subset of the weights of the first layer of the regression
    subnetwork (assumed to start with vgg_16_regression) with the pre-trained
    ILSVRC weights.

    The rest of the first layer gets zero-initialized.

    I.e. subnetwork[:, :, 0:3, :] <- ILSVRC weights
         subnetwork[:, :, 3:16, :] <- Zeros
    """
    first_layer_vgg16 = tf.Variable(initial_value=tf.zeros([3, 3, 3, 64]))

    first_layer_restorer = tf.train.Saver(var_list={'vgg_16/conv1/conv1_1/weights': first_layer_vgg16})
    first_layer_restorer.restore(sess=session, save_path=second_checkpoint_path)

    conv1_initializer = tf.concat(values=[first_layer_vgg16, tf.zeros(shape=[3, 3, 16, 64])],
                                  axis=2)
    regression_subnet_conv1 = next(var for var in tf.global_variables()
                                   if 'vgg_16_regression/conv1/conv1_1/weights' in var.op.name)
    session.run(
        tf.assign(ref=regression_subnet_conv1, value=conv1_initializer))


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


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


def _init_df():
    if FLAGS.restore_global_step == False:
        config = tf.flags.FLAGS.__flags
        checkpoint_name = 'model'
        runtime_dict = {'Checkpoint_Name':checkpoint_name,
                        'Learning_Rate':[],
                        'Epoch':[],
                        'Mean_Training_Loss':[],
                        'Mean_Validation_Loss':[],
                        'Total PCKh':[]
                        }
        runtime_df = pd.DataFrame(runtime_dict)
        log_dict = merge_two_dicts(config, runtime_dict)
        log_df = pd.DataFrame(log_dict)
    else:
        config = tf.flags.FLAGS.__flags
        print(config)
        log_df = pd.read_hdf(FLAGS.log_dir + '/' +
                             FLAGS.checkpoint_name + '.h5')
        # Get sub-dataframe for runtime
        runtime_cols = [log_df[joint] for joint in JOINT_NAMES]
        runtime_cols.append(log_df.Learning_Rate,
                            log_df.Epoch,
                            log_df.Mean_Training_Loss,
                            log_df.Mean_Validation_Loss)

        runtime_df = log_df(runtime_cols)

    return log_df, runtime_df


def _save_df(log_df):
    """Saves dataframe to hdf"""
    log_df.to_hdf(FLAGS.checkpoint_name + '.h5')


def _update_(log_df, runtime_df):
    log_df = pd.merge(log_df, runtime_df)
    stop_early = False
    return log_df, stop_early

def train():
    """Trains a human pose estimation network to detect and/or regress binary
    maps and/or confidence maps of joint co-ordinates (NUM_JOINTS sets of (x,
    y) co-ordinates).

    Here we only take files 0-N.
    For now files are manually renamed as train[0-N].tfrecord and
    valid[N-M].tfrecord, where there are N + 1 train records, (M - N)
    validation records and M + 1 records in total.
    """
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            #pdb.set_trace()
            num_batches_per_epoch, train_op, train_loss, global_step = _setup_training(FLAGS)

            eval_graph = tf.Graph()
            with eval_graph.as_default():
                num_val_examples, val_loss, val_logits, gt_data = setup_evaluation(FLAGS)

            session = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True))
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())

            if FLAGS.is_regression_subnetwork_pretrained:
                _init_regression_subnetwork_first_layer(session,
                                                        FLAGS.second_checkpoint_path)
            # This check for global_step
            _restore_checkpoint_variables(session,
                                          global_step,
                                          FLAGS.checkpoint_path,
                                          FLAGS.checkpoint_exclude_scopes,
                                          FLAGS.is_regression_subnetwork_pretrained)

            if FLAGS.second_checkpoint_path is not None:
                _restore_checkpoint_variables(session,
                                              global_step,
                                              FLAGS.second_checkpoint_path,
                                              FLAGS.second_checkpoint_exclude_scopes,
                                              FLAGS.is_regression_subnetwork_pretrained)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            train_writer = tf.summary.FileWriter(
                logdir=FLAGS.log_dir,
                graph=session.graph)

            # Human Edible Report
            log_handle = open(os.path.join(FLAGS.log_dir, FLAGS.log_filename), 'a')
            # For Pandas overviewer we fill this dict with data at every epoch
            # and update.
            # The _init_df function checks for existence and merges if
            # FLAGS.restore_global_step is True
            log_df, runtime_df = _init_df()
            saver = tf.train.Saver(var_list=tf.global_variables())

            with eval_graph.as_default():
                restorer = tf.train.Saver(var_list=tf.global_variables())

            summary_op = tf.summary.merge_all()

            epoch = 0
            while epoch < FLAGS.max_epochs:
                runtime_df = _train_single_epoch(session,
                                                 saver,
                                                 train_writer,
                                                 train_op,
                                                 train_loss,
                                                 global_step,
                                                 summary_op,
                                                 num_batches_per_epoch,
                                                 log_handle,
                                                 FLAGS.log_dir,
                                                 runtime_df)

                runtime_df['Epoch'].append(epoch)
                runtime_df['Mean_Training_Loss'].append(train_epoch_mean_loss)
                with eval_graph.as_default():
                    runtime_df = evaluate_single_epoch(restorer,
                                                       FLAGS.log_dir,
                                                       val_loss,
                                                       val_logits,
                                                       gt_data,
                                                       num_val_examples,
                                                       FLAGS.batch_size,
                                                       FLAGS.image_dim,
                                                       epoch,
                                                       log_handle,
                                                       runtime_df)

                    log_df, stop_early = _update_df(log_df, runtime_df)
                    if stop_early:
                        _save_df(log_df)
                        break

            log_handle.close()
            train_writer.close()
            coord.request_stop()
            coord.join(threads=threads)


def main(argv=None):
    """Usage: python3 -m train
    (After running write_tf_record.py. See its docstring for usage.)

    See top of this file for flags, e.g. --log_dir=./log, or type
    'python3 -m train --help' for options.
    """
    train()


if __name__ == "__main__":
    tf.app.run()
