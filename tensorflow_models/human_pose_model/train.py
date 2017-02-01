import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from logging import INFO
from nets import NETS, NET_ARG_SCOPES, NET_LOSS
from mpii_read import Person
from input_pipeline import setup_train_input_pipeline
from evaluate import evaluate

FLAGS = tf.app.flags.FLAGS

# NOTE(brendan): equal to the number of joint-annotated people per file
# containing MPII Human Pose Dataset training examples.
NUM_EXAMPLES_PER_SHARD = 953

RMSPROP_DECAY = 0.9
RMSPROP_MOMENTUM = 0.9
RMSPROP_EPSILON = 1.0

NUM_JOINTS = Person.NUM_JOINTS

tf.app.flags.DEFINE_string('network_name', None,
                           """Name of desired network to use for part
                           detection. Valid options: vgg, inception_v3.""")
tf.app.flags.DEFINE_string('data_dir', './train',
                           """Path to take input TFRecord files from.""")
tf.app.flags.DEFINE_string('log_dir', './log',
                           """Path to take summaries and checkpoints from, and
                           write them to.""")
tf.app.flags.DEFINE_string('log_filename', 'train_log',
                           """Name of file to log training steps and loss
                           to.""")
tf.app.flags.DEFINE_string('checkpoint_path', None,
                           """Path to take checkpoint file (e.g.
                           inception_v3.ckpt) from.""")
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None,
                           """Comma-separated list of scopes to exclude when
                           restoring from a checkpoint.""")
tf.app.flags.DEFINE_string('trainable_scopes', None,
                           """Comma-separated list of scopes to train.""")

tf.app.flags.DEFINE_integer('image_dim', 299,
                            """Dimension of the square input image.""")

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of threads to use to preprocess
                            images.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of threads to use to read example
                            protobufs from TFRecords.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """Number of GPUs in system.""")

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Size of each mini-batch (number of examples
                            processed at once).""")
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Factor by which to increase the minimum examples
                            in RandomShuffleQueue.""")
tf.app.flags.DEFINE_integer('max_epochs', 100,
                            """Maximum number of epochs in training run.""")

tf.app.flags.DEFINE_integer('heatmap_stddev_pixels', 5,
                            """Standard deviation of Gaussian joint heatmap, in
                            pixels.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Rate at which learning rate is decayed.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Number of epochs before decay factor is applied
                          once.""")

tf.app.flags.DEFINE_boolean('restore_global_step', False,
                            """Set to True if restoring a training run that is
                            part-way complete.""")

def _summarize_inception_model(endpoints):
    """Summarizes the activation values that are marked for summaries in the
    Inception v3 network.
    """
    with tf.name_scope('summaries'):
        for activation in endpoints.values():
            tensor_name = activation.op.name
            tf.summary.histogram(
                name=tensor_name + '/activations',
                values=activation)
            tf.summary.scalar(
                name=tensor_name + '/sparsity',
                tensor=tf.nn.zero_fraction(value=activation))


def _inference(training_batch):
    """Sets up a human pose inference model, computes predictions on input
    images and calculates loss on those predictions based on an input dense
    vector of joint location confidence maps and binary maps (the ground truth
    vector).

    TF-slim's `arg_scope` is used to keep variables (`slim.model_variable`) in
    CPU memory. See the training procedure block diagram in the TF Inception
    [README](https://github.com/tensorflow/models/tree/master/inception).

    Args:
        training_batch: A batch of training images with associated joint
            vectors.

    Returns:
        Tensor giving the total loss (combined loss from auxiliary and primary
        logits, added to regularization losses).
    """
    part_detect_net = NETS[FLAGS.network_name]
    net_arg_scope = NET_ARG_SCOPES[FLAGS.network_name]
    net_loss = NET_LOSS[FLAGS.network_name]

    with slim.arg_scope([slim.model_variable], device='/cpu:0'):
        with slim.arg_scope(net_arg_scope()):
            logits, endpoints = part_detect_net(inputs=training_batch.images,
                                                num_classes=NUM_JOINTS)

            net_loss(logits,
                     endpoints,
                     training_batch.heatmaps,
                     training_batch.weights)

            # TODO(brendan): Calculate loss averages for tensorboard

            total_loss = slim.losses.get_total_loss()

    return total_loss


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


def _setup_training_op(training_batch, global_step, optimizer):
    """Sets up inference (predictions), loss calculation, and minimization
    based on the input optimizer.

    Args:
        training_batch: Batch of preprocessed examples dequeued from the input
            pipeline.
        global_step: Training step counter.
        optimizer: Optimizer to minimize the loss function.

    Returns: Operation to run a training step.
    """
    with tf.device(device_name_or_function='/gpu:0'):
        loss = _inference(training_batch)

        train_op = slim.learning.create_train_op(
            total_loss=loss,
            optimizer=optimizer,
            global_step=global_step,
            variables_to_train=_get_variables_to_train())

        tf.summary.scalar(name='loss', tensor=loss)

    return train_op


def _restore_checkpoint_variables(session, global_step):
    """Initializes the model in the graph of a passed session with the
    variables in the file found in `FLAGS.checkpoint_path`, except those
    excluded by `FLAGS.checkpoint_exclude_scopes`.
    """
    if FLAGS.checkpoint_path is None:
        return

    if FLAGS.checkpoint_exclude_scopes is None:
        variables_to_restore = slim.get_model_variables()
    else:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

    if FLAGS.restore_global_step:
        variables_to_restore.append(global_step)

    restorer = tf.train.Saver(var_list=variables_to_restore)
    restorer.restore(sess=session, save_path=FLAGS.checkpoint_path)


def train():
    """Trains an Inception v3 network to regress joint co-ordinates (NUM_JOINTS
    sets of (x, y) co-ordinates) directly.
    """
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            # TODO(brendan): Support multiple GPUs?
            assert FLAGS.num_gpus == 1

            data_filenames = tf.gfile.Glob(
                os.path.join(FLAGS.data_dir, 'train*tfrecord'))
            assert data_filenames, ('No data files found.')
            assert len(data_filenames) >= FLAGS.num_readers

            training_batch = setup_train_input_pipeline(
                FLAGS.num_readers,
                FLAGS.input_queue_memory_factor,
                FLAGS.batch_size,
                FLAGS.num_preprocess_threads,
                FLAGS.image_dim,
                FLAGS.heatmap_stddev_pixels,
                data_filenames)

            examples_per_epoch = (NUM_EXAMPLES_PER_SHARD * len(data_filenames))
            num_batches_per_epoch = int(examples_per_epoch / FLAGS.batch_size)
            global_step, optimizer = _setup_optimizer(num_batches_per_epoch,
                                                      FLAGS.num_epochs_per_decay,
                                                      FLAGS.initial_learning_rate,
                                                      FLAGS.learning_rate_decay_factor)

            train_op = _setup_training_op(training_batch,
                                          global_step,
                                          optimizer)

            # TODO(brendan): track moving averages of trainable variables

            log_handle = open(os.path.join(FLAGS.log_dir, FLAGS.log_filename),
                              'a')

            saver = tf.train.Saver(tf.global_variables())

            summary_op = tf.summary.merge_all()

            init = tf.global_variables_initializer()

            session = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True))
            session.run(init)

            _restore_checkpoint_variables(session, global_step)

            tf.train.start_queue_runners(sess=session)

            summary_writer = tf.summary.FileWriter(
                logdir=FLAGS.log_dir,
                graph=session.graph)

            epoch = 0
            while epoch < FLAGS.max_epochs:
                for batch_step in range(num_batches_per_epoch):
                    start_time = time.time()
                    batch_loss, total_steps = session.run(fetches=[train_op, global_step])
                    duration = time.time() - start_time

                    assert not np.isnan(batch_loss)

                    if (total_steps % 100) == 0:
                        log_handle.write('step {}: loss = {} ({:.2f} sec/step)\n'
                                         .format(total_steps, batch_loss, duration))
                        log_handle.flush()

                        summary_str = session.run(summary_op)
                        summary_writer.add_summary(summary=summary_str,
                                                   global_step=total_steps)

                    if ((total_steps % 1000) == 0):
                        checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                        saver.save(sess=session, save_path=checkpoint_path, global_step=total_steps)

                epoch = int(total_steps/examples_per_epoch)
                log_handle.write('Epoch {} done. Evaluating metrics.\n'.format(epoch))
                log_handle.flush()

                latest_checkpoint = tf.train.latest_checkpoint(
                    checkpoint_dir=FLAGS.log_dir)
                evaluate(FLAGS.network_name,
                         FLAGS.data_dir,
                         latest_checkpoint,
                         os.path.join(FLAGS.log_dir, 'eval_log'),
                         FLAGS.image_dim,
                         FLAGS.num_preprocess_threads,
                         FLAGS.batch_size,
                         epoch,
                         NUM_EXAMPLES_PER_SHARD)

            log_handle.close()


def main(argv=None):
    """Usage: python3 -m train
    (After running write_tf_record.py. See its docstring for usage.)

    See top of this file for flags, e.g. --log_dir=./log, or type
    'python3 -m train --help' for options.
    """
    train()


if __name__ == "__main__":
    tf.app.run()
