import os
import time
import copy
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
import mpii_read

FLAGS = tf.app.flags.FLAGS

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 18000

RMSPROP_DECAY = 0.9
RMSPROP_MOMENTUM = 0.9
RMSPROP_EPSILON = 1.0

NUM_JOINTS = 16
NUM_JOINT_COORDS = 2*NUM_JOINTS

tf.app.flags.DEFINE_string('data_dir', '.',
                           """Path to take input TFRecord files from.""")

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

tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Rate at which learning rate is decayed.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Number of epochs before decay factor is applied
                          once.""")

class TrainingBatch(object):
    """Contains a training batch of images along with corresponding
    ground-truth joint vectors for the annotated person in that image.

    images, joints and joint_indices should all be lists of length
    `FLAGS.batch_size`.
    """
    def __init__(self, images, joints, joint_indices):
        self._images = images
        self._joints = joints
        self._joint_indices = joint_indices

    @property
    def images(self):
        return self._images

    @property
    def joints(self):
        return self._joints

    @property
    def joint_indices(self):
        return self._joint_indices


def _setup_example_queue(filename_queue,
                         num_readers,
                         input_queue_memory_factor,
                         batch_size):
    """Sets up a randomly shuffled queue containing example protobufs, read
    from the TFRecord files in `filename_queue`.

    Args:
        filename_queue: A queue of filepaths to the TFRecord files containing
            the input Example protobufs.
        num_readers: Number of file readers to use.
        input_queue_memory_factor: Factor by which to scale up the minimum
            number of examples in the example queue. A larger factor increases
            the mixing of examples, but will also increase memory pressure.
        batch_size: Number of training elements in a batch.

    Returns:
        A dequeue op that will dequeue one Tensor containing an input example
        from `examples_queue`.
    """
    # TODO(brendan): Alter `write_tf_record` code to spit out
    # shards with about 1024 examples each.
    examples_per_shard = 1024
    min_queue_examples = input_queue_memory_factor*examples_per_shard

    examples_queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3*batch_size,
                                           min_after_dequeue=min_queue_examples,
                                           dtypes=[tf.string])

    enqueue_ops = []
    for _ in range(num_readers):
        reader = tf.TFRecordReader()
        _, per_thread_example = reader.read(queue=filename_queue)
        enqueue_ops.append(examples_queue.enqueue(vals=[per_thread_example]))

    tf.train.queue_runner.add_queue_runner(
        qr=tf.train.queue_runner.QueueRunner(queue=examples_queue,
                                             enqueue_ops=enqueue_ops))

    return examples_queue.dequeue()


def _parse_and_preprocess_images(example_serialized,
                                 num_preprocess_threads,
                                 image_dim):
    """Parses Example protobufs containing input images and their ground truth
    vectors and preprocesses those images, returning a vector with one
    preprocessed tensor per thread.

    The loop over the threads (as opposed to a deep copy of the first resultant
    Tensor) allows for different image distortion to be done depending on the
    thread ID, although currently all image preprocessing is identical.

    Args:
        example_serialized: Tensor containing a serialized example, as read
            from a TFRecord file.
        num_preprocess_threads: Number of threads to use for image
            preprocessing.
        image_dim: Dimension of square input images.

    Returns:
        A list of lists, one for each thread, where each inner list contains a
        decoded image with colours scaled to range [-1, 1], as well as the
        sparse joint ground truth vectors.
    """
    images_and_joints = []
    for thread_id in range(num_preprocess_threads):
        feature_map = {
            'image_jpeg': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'joint_indices': tf.VarLenFeature(dtype=tf.int64),
            'joints': tf.VarLenFeature(dtype=tf.float32)
        }
        features = tf.parse_single_example(
            serialized=example_serialized, features=feature_map)

        img_jpeg = features['image_jpeg']
        with tf.name_scope(name='decode_jpeg', values=[img_jpeg]):
            img_tensor = tf.image.decode_jpeg(contents=img_jpeg,
                                              channels=3)
            decoded_img = tf.image.convert_image_dtype(
                image=img_tensor, dtype=tf.float32)

        # TODO(brendan): Image distortion goes here.

        decoded_img = tf.sub(x=decoded_img, y=0.5)
        decoded_img = tf.mul(x=decoded_img, y=2.0)

        decoded_img = tf.reshape(
            tensor=decoded_img,
            shape=[image_dim, image_dim, 3])

        images_and_joints.append([decoded_img,
                                  features['joint_indices'],
                                  features['joints']])

    return images_and_joints


def _setup_input_pipeline(data_dir,
                          num_readers,
                          input_queue_memory_factor,
                          batch_size,
                          num_preprocess_threads,
                          image_dim):
    """Sets up an input pipeline that reads example protobufs from all TFRecord
    files, assumed to be named train*.tfrecord (e.g. train0.tfrecord),
    decodes and preprocesses the images.

    There are three queues: `filename_queue`, contains the TFRecord filenames,
    and feeds `examples_queue` with serialized example protobufs.

    Then, serialized examples are dequeued from `examples_queue`, preprocessed
    in parallel by `num_preprocess_threads` and the result is enqueued into the
    queue created by `batch_join`. A dequeue operation from the `batch_join`
    queue is what is returned from `preprocess_images`. What is dequeued is a
    batch of size `batch_size` containing a set of, for example, 32 images in
    the case of `images` or a sparse vector of floating-point joint
    co-ordinates (in range [0,1]) in the case of `joints`.

    Also adds a summary for the images.

    Args:
        data_dir: Path to take input TFRecord files from.
        num_readers: Number of file readers to use.
        input_queue_memory_factor: Input to `_setup_example_queue`. See
            `_setup_example_queue` for details.
        batch_size: Number of examples to process at once (in one training
            step).
        num_preprocess_threads: Number of threads to use to preprocess image
            data.
        image_dim: Dimension of square input images.

    Returns:
        TrainingBatch(images, joints, joint_indices): Lists of image tensors,
            sparse joints (ground truth vectors), and sparse joint indices,
            each of shape [batch_size]
    """
    # TODO(brendan): num_readers == 1 case
    assert num_readers > 1

    with tf.name_scope('batch_processing'):
        data_filenames = tf.gfile.Glob(os.path.join(data_dir, 'train*tfrecord'))
        assert data_filenames, ('No data files found.')
        assert len(data_filenames) >= num_readers

        filename_queue = tf.train.string_input_producer(
            string_tensor=data_filenames,
            capacity=16)

        example_serialized = _setup_example_queue(filename_queue,
                                                  num_readers,
                                                  input_queue_memory_factor,
                                                  batch_size)

        images_and_joints = _parse_and_preprocess_images(
            example_serialized, num_preprocess_threads, image_dim)

        images, joint_indices, joints = tf.train.batch_join(
            tensors_list=images_and_joints,
            batch_size=batch_size,
            capacity=2*num_preprocess_threads*batch_size)

        tf.summary.image(name='images', tensor=images)

        return TrainingBatch(images, joints, joint_indices)


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


def _sparse_joints_to_dense(joints,
                            joint_indices,
                            batch_size,
                            num_joint_coords):
    """Converts a sparse vector of joints to a dense format, and also returns a
    set of weights indicating which joints are present.

    Args:
        joints: Sparse vector of joints.
        joint_indices: Sparse indices of the joints.
        batch_size: Number of training elements in a batch.
        num_joint_coords: Total number of joint co-ordinates, where (x0, y0)
            counts as two co-ordinates.

    Returns:
        (dense_joints, weights) tuple, where dense_joints is a dense vector of
        shape [batch_size, num_joint_coords], with zeros in the indices not
        present in the sparse vector. `weights` contains 1s for all the present
        joints and 0s otherwise.
    """
    sparse_joints = tf.sparse_merge(sp_ids=joint_indices,
                                    sp_values=joints,
                                    vocab_size=num_joint_coords)
    dense_joints = tf.sparse_tensor_to_dense(sp_input=sparse_joints,
                                             default_value=0)

    dense_shape = [batch_size, num_joint_coords]
    dense_joints = tf.reshape(tensor=dense_joints, shape=dense_shape)

    weights = tf.sparse_to_dense(sparse_indices=sparse_joints.indices,
                                 output_shape=dense_shape,
                                 sparse_values=1,
                                 default_value=0)

    return dense_joints, weights


def _inference(training_batch, batch_size, num_joint_coords, scope):
    """Sets up an Inception v3 model, computes predictions on input images and
    calculates loss on those predictions based on an input sparse vector of
    joints (the ground truth vector).

    TF-slim's `arg_scope` is used to keep variables (`slim.model_variable`) in
    CPU memory. See the training procedure block diagram in the TF Inception
    [README](https://github.com/tensorflow/models/tree/master/inception).

    Args:
        training_batch: A batch of training images with associated joint
            vectors.
        batch_size: Number of training elements in a batch.
        num_joint_coords: Total number of joint co-ordinates, where (x0, y0)
            counts as two co-ordinates.
        scope: The name scope (for summaries, debugging).

    Returns:
        Tensor giving the total loss (combined loss from auxiliary and primary
        logits, added to regularization losses).
    """
    with slim.arg_scope([slim.model_variable], device='/cpu:0'):
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, endpoints = inception.inception_v3(inputs=training_batch.images,
                                                       num_classes=num_joint_coords,
                                                       scope=scope)

            _summarize_inception_model(endpoints)

            dense_joints, weights = _sparse_joints_to_dense(
                training_batch.joints,
                training_batch.joint_indices,
                batch_size,
                num_joint_coords)

            auxiliary_logits = endpoints['AuxLogits']

            slim.losses.mean_squared_error(predictions=logits,
                                           labels=dense_joints,
                                           weights=weights)
            slim.losses.mean_squared_error(predictions=auxiliary_logits,
                                           labels=dense_joints,
                                           weights=weights,
                                           scope='aux_logits')

            # TODO(brendan): Calculate loss averages for tensorboard

            total_loss = slim.losses.get_total_loss()

    return total_loss


def main(argv=None):
    """Usage: python3 -m read_tf_record
    (After running write_tf_record.py. See its docstring for usage.)
    """
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            # TODO(brendan): Support multiple GPUs?
            assert FLAGS.num_gpus == 1

            training_batch = _setup_input_pipeline(FLAGS.data_dir,
                                                   FLAGS.num_readers,
                                                   FLAGS.input_queue_memory_factor,
                                                   FLAGS.batch_size,
                                                   FLAGS.num_preprocess_threads,
                                                   FLAGS.image_dim)

            input_summaries = copy.copy(tf.get_collection(key=tf.GraphKeys.SUMMARIES))

            global_step = tf.get_variable(
                name='global_step',
                shape=[],
                initializer=tf.constant_initializer(0),
                trainable=False)

            num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                     FLAGS.batch_size)
            decay_steps = int(num_batches_per_epoch*FLAGS.num_epochs_per_decay)
            learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.initial_learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=decay_steps,
                                                       decay_rate=FLAGS.learning_rate_decay_factor)

            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                  decay=RMSPROP_DECAY,
                                                  momentum=RMSPROP_MOMENTUM,
                                                  epsilon=RMSPROP_EPSILON)

            with tf.device(device_name_or_function='/gpu:0'):
                with tf.name_scope(name='tower0') as scope:
                    loss = _inference(training_batch,
                                      FLAGS.batch_size,
                                      NUM_JOINT_COORDS,
                                      scope)

                    summaries = tf.get_collection(key=tf.GraphKeys.SUMMARIES, scope=scope)

                    batchnorm_updates = tf.get_collection(
                        key=tf.GraphKeys.UPDATE_OPS, scope=scope)

                    gradients = optimizer.compute_gradients(loss=loss)

            summaries.extend(input_summaries)

            summaries.append(
                tf.summary.scalar(name='learning_rate', tensor=learning_rate))

            for grad, var in gradients:
                if grad is not None:
                    summaries.append(
                        tf.summary.histogram(name=var.op.name + '/gradients',
                                             values=grad))

            apply_gradient_op = optimizer.apply_gradients(
                grads_and_vars=gradients, global_step=global_step)

            for var in tf.trainable_variables():
                summaries.append(
                    tf.summary.histogram(name=var.op.name, values=var))

            # TODO(brendan): track moving averages of trainable variables

            batchnorm_updates_op = tf.group(*batchnorm_updates)
            train_op = tf.group(batchnorm_updates_op, apply_gradient_op)

            saver = tf.train.Saver(var_list=tf.global_variables())

            summary_op = tf.summary.merge(inputs=summaries)

            init = tf.global_variables_initializer()

            session = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True))
            session.run(fetches=init)

            # TODO(brendan): check for pre-trained checkpoint

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            log_dir = '../../log'
            summary_writer = tf.summary.FileWriter(logdir=log_dir,
                                                   graph=session.graph)

            for step in range(1000000):
                start = time.perf_counter()
                _, loss_value = session.run(fetches=[train_op, loss])
                end = time.perf_counter()
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    print('step {}, loss = {}'.format(step, loss_value))

                if step % 100 == 0:
                    summary_string = session.run(fetches=summary_op)
                    summary_writer.add_summary(summary=summary_string)

                if step % 5000 == 0:
                    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess=session,
                               save_path=checkpoint_path,
                               global_step=step)

            coord.request_stop()
            coord.join(threads=threads)


if __name__ == "__main__":
    tf.app.run()
