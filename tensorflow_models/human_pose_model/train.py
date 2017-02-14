import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from logging import INFO
from nets import NETS, NET_ARG_SCOPES, NET_LOSS
from mpii_read import Person
from input_pipeline import setup_train_input_pipeline
from evaluate import evaluate
from tqdm import trange
import pdb

FLAGS = tf.app.flags.FLAGS

RMSPROP_DECAY = 0.9
RMSPROP_MOMENTUM = 0.9
RMSPROP_EPSILON = 1.0

NUM_JOINTS = Person.NUM_JOINTS

tf.app.flags.DEFINE_string('network_name', 'vgg_bulat',
                           """Name of desired network to use for part
                           detection. Valid options: vgg, inception_v3.""")

tf.app.flags.DEFINE_string('train_data_dir', './train_vgg_fcn',
                           """Path to take input training TFRecord files
                           from.""")

tf.app.flags.DEFINE_string('validation_data_dir', './valid_vgg_fcn',
                           """Path to take input validation TFRecord files
                           from.""")

tf.app.flags.DEFINE_string('log_dir', '/mnt/data/datasets/MPII_HumanPose/logs/vgg_bulat/lr1e-4_bs16',
                           """Path to take summaries and checkpoints from, and
                           write them to.""")

tf.app.flags.DEFINE_string('log_filename', 'train_log',
                           """Name of file to log training steps and loss
                           to.""")

tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints/vgg_16.ckpt',
                           """Path to take checkpoint file (e.g.
                           inception_v3.ckpt) from.""")

tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None,
                           """Comma-separated list of scopes to exclude when
                           restoring from a checkpoint.""")

tf.app.flags.DEFINE_string('trainable_scopes', None,
                           """Comma-separated list of scopes to train.""")

tf.app.flags.DEFINE_integer('image_dim', 380,
                            """Dimension of the square input image.""")

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of threads to use to preprocess
                            images.""")

tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of threads to use to read example
                            protobufs from TFRecords.""")

tf.app.flags.DEFINE_integer('num_gpus', 3,
                            """Number of GPUs in system.""")

tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Size of each mini-batch (number of examples
                            processed at once).""")

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Factor by which to increase the minimum examples
                            in RandomShuffleQueue.""")

tf.app.flags.DEFINE_integer('max_epochs', 30,
                            """Maximum number of epochs in training run.""")

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

tf.app.flags.DEFINE_integer('heatmap_stddev_pixels',5,
                            """Standard deviation of Gaussian joint heatmap, in pixels.""")

def _summarize_bulat_model(endpoints):
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


def _summarize_loss(total_loss, gpu_index):
    """Summarizes the loss and average loss for this tower, and ensures that
    loss averages are computed every time the loss is computed.
    """
    loss_averages = tf.train.ExponentialMovingAverage(
        decay=0.9, name='avg')
    loss_averages_op = loss_averages.apply(var_list=[total_loss])

    tf.summary.scalar(name='tower_{}_loss'.format(gpu_index),
                      tensor=total_loss)
    tf.summary.scalar(name='tower_{}_loss_avg'.format(gpu_index),
                      tensor=loss_averages.average(total_loss))

    with tf.control_dependencies(control_inputs=[loss_averages_op]):
        total_loss = tf.identity(total_loss)

    return total_loss


def _inference(images, heatmaps, weights, gpu_index, scope):
    """Sets up a human pose inference model, computes predictions on input
    images and calculates loss on those predictions based on an input dense
    vector of joint location confidence maps and binary maps (the ground truth
    vector).

    TF-slim's `arg_scope` is used to keep variables (`slim.model_variable`) in
    CPU memory. See the training procedure block diagram in the TF Inception
    [README](https://github.com/tensorflow/models/tree/master/inception).

    Args:
        images: Mini-batch of preprocessed examples dequeued from the input
            pipeline.
        heatmaps: Confidence maps of ground truth joints.
        weights: Weights of heatmaps (tensors of all 1s if joint present, all
            0s if not present).
        gpu_index: Index of GPU calculating the current loss.
        scope: Name scope for ops, which is different for each tower (tower_N).

    Returns:
        Tensor giving the total loss (combined loss from auxiliary and primary
        logits, added to regularization losses).
    """
    part_detect_net = NETS[FLAGS.network_name]
    net_arg_scope = NET_ARG_SCOPES[FLAGS.network_name]
    net_loss = NET_LOSS[FLAGS.network_name]
    with slim.arg_scope([slim.model_variable], device='/cpu:0'):
        with slim.arg_scope(net_arg_scope()):
            with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=(gpu_index > 0)):
                logits, endpoints = part_detect_net(inputs=images,
                                                    num_classes=NUM_JOINTS,
                                                    scope=scope)


            net_loss(logits, endpoints, heatmaps, weights)

            losses = tf.get_collection(key=tf.GraphKeys.LOSSES, scope=scope)
            regularization_losses = tf.get_collection(key=tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
            total_loss = tf.add_n(inputs=losses + regularization_losses, name='total_loss')

            merged_logits = tf.reshape(
                tf.reduce_max(logits, 3),
                [int(images.get_shape()[0]), FLAGS.image_dim, FLAGS.image_dim, 1])
            merged_logits = tf.cast(merged_logits, tf.float32)
            tf.summary.image(name='logits', tensor=merged_logits)

            total_loss = _summarize_loss(total_loss, gpu_index)

    return total_loss, logits


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

        grad = tf.concat(concat_dim=0, values=grads)
        grad = tf.reduce_mean(input_tensor=grad, axis=0)

        avg_grad_and_vars.append((grad, grad_and_vars[0][1]))

    return avg_grad_and_vars


def _setup_training_op(images,
                       heatmaps,
                       weights,
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
    images_split = tf.split(split_dim=0, num_split=num_gpus, value=images)
    heatmaps_split = tf.split(split_dim=0, num_split=num_gpus, value=heatmaps)
    weights_split = tf.split(split_dim=0, num_split=num_gpus, value=weights)

    tower_grads = []
    for gpu_index in range(num_gpus):
        with tf.device(device_name_or_function='/gpu:{}'.format(gpu_index)):
            with tf.name_scope('tower_{}'.format(gpu_index)) as scope:
                total_loss,_ = _inference(images_split[gpu_index],
                                        heatmaps_split[gpu_index],
                                        weights_split[gpu_index],
                                        gpu_index,
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


def _restore_checkpoint_variables(session, global_step):
    """Initializes the model in the graph of a passed session with the
    variables in the file found in `FLAGS.checkpoint_path`, except those
    excluded by `FLAGS.checkpoint_exclude_scopes`.
    """
    pdb.set_trace()
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
                # For some reason there are no biases in the resnet checkpoint 0_o
                if FLAGS.network_name == 'resnet_bulat' and var.op.name.endswith('biases'):
                    excluded = True
                    break
                # There are no batch norm layers in the original vgg - Not having regexp is so annoying
                if FLAGS.network_name == 'vgg_bulat_bn_relu' and var.op.name.endswith('BatchNorm/moving_mean'):
                    print('i am here')
                    excluded = True
                    break
                if FLAGS.network_name == 'vgg_bulat_bn_relu' and var.op.name.endswith('BatchNorm/moving_variance'):
                    print('i am here')
                    excluded = True
                    break
                if FLAGS.network_name == 'vgg_bulat_bn_relu' and var.op.name.endswith('BatchNorm/beta'):
                    print('i am here')
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

    Here we only take files 0-N.
    For now files are manually renamed as train[0-N].tfrecord and
    valid[N-M].tfrecord, where there are N + 1 train records, (M - N)
    validation records and M + 1 records in total.
    """
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            train_data_filenames = tf.gfile.Glob(
                os.path.join(FLAGS.train_data_dir, 'train*tfrecord'))
            assert train_data_filenames, ('No data files found.')
            assert len(train_data_filenames) >= FLAGS.num_readers

            num_training_examples = 0
            options = tf.python_io.TFRecordOptions(
                compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
            for data_file in train_data_filenames:
                for _ in tf.python_io.tf_record_iterator(path=data_file, options=options):
                    num_training_examples += 1

            # Merged with FLAGS
            images, binary_maps, heatmaps, weights = setup_train_input_pipeline(
                FLAGS, train_data_filenames)

            num_batches_per_epoch = int(num_training_examples / FLAGS.batch_size)
            global_step, optimizer = _setup_optimizer(num_batches_per_epoch,
                                                      FLAGS.num_epochs_per_decay,
                                                      FLAGS.initial_learning_rate,
                                                      FLAGS.learning_rate_decay_factor)

            train_op, loss = _setup_training_op(images,
                                                heatmaps,
                                                weights,
                                                global_step,
                                                optimizer,
                                                FLAGS.num_gpus)

            log_handle = open(os.path.join(FLAGS.log_dir, FLAGS.log_filename),'a')

            saver = tf.train.Saver(tf.global_variables())

            summary_op = tf.summary.merge_all()

            init = tf.global_variables_initializer()

            session = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True))
            session.run(init)

            _restore_checkpoint_variables(session, global_step)

            tf.train.start_queue_runners(sess=session)

            train_writer = tf.summary.FileWriter(
                logdir=FLAGS.log_dir,
                graph=session.graph)

            epoch = 0
            # To follow mean loss over epoch
            curr_step = 0
            while epoch < FLAGS.max_epochs:
                train_epoch_mean_loss = 0
                Epoch = trange(num_batches_per_epoch, desc='Loss', leave=True)
                for batch_step in Epoch:
                    start_time = time.time()
                    _, batch_loss, total_steps = session.run(
                        fetches=[train_op, loss, global_step])
                    duration = time.time() - start_time

                    desc = ('step {}: loss = {} ({:.2f} sec/step)'
                            .format(total_steps, batch_loss, duration))
                    train_epoch_mean_loss += batch_loss
                    Epoch.set_description(desc)
                    Epoch.refresh()
                    assert not np.isnan(batch_loss)
                    if (total_steps % 100) == 0:
                        log_handle.write(desc + '\n')
                        log_handle.flush()

                        summary_str = session.run(summary_op)
                        train_writer.add_summary(summary=summary_str,
                                                 global_step=total_steps)

                checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess=session, save_path=checkpoint_path, global_step=total_steps)

                epoch = int(total_steps/num_batches_per_epoch)
                train_epoch_mean_loss /= (total_steps - curr_step)
                curr_step = total_steps
                log_handle.write('Epoch {} done.\n'.format(epoch))
                log_handle.write('Mean training loss is {}.\n'.format(train_epoch_mean_loss))
                log_handle.flush()

                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=FLAGS.log_dir)
                assert latest_checkpoint is not None

                evaluate(FLAGS.network_name,
                         FLAGS.validation_data_dir,
                         latest_checkpoint,
                         os.path.join(FLAGS.log_dir, 'eval_log'),
                         FLAGS.image_dim,
                         FLAGS.num_preprocess_threads,
                         FLAGS.batch_size,
                         FLAGS.num_gpus,
                         FLAGS.heatmap_stddev_pixels,
                         epoch)

            log_handle.close()
            train_writer.close()


def main(argv=None):
    """Usage: python3 -m train
    (After running write_tf_record.py. See its docstring for usage.)

    See top of this file for flags, e.g. --log_dir=./log, or type
    'python3 -m train --help' for options.
    """
    train()


if __name__ == "__main__":
    tf.app.run()
