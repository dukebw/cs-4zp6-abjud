import os
import tensorflow as tf

# TODO(brendan): Alter `write_tf_record` code to spit out
# shards with about 1024 examples each.
EXAMPLES_PER_SHARD = 1024

class TrainingBatch(object):
    """Contains a training batch of images along with corresponding
    ground-truth joint vectors for the annotated person in that image.

    images, joints and joint_indices should all be lists of length
    `batch_size`.
    """
    def __init__(self, images, joints, joint_indices, batch_size):
        assert images.get_shape()[0] == batch_size
        self._images = images
        self._joints = joints
        self._joint_indices = joint_indices
        self._batch_size = batch_size

    @property
    def images(self):
        return self._images

    @property
    def joints(self):
        return self._joints

    @property
    def joint_indices(self):
        return self._joint_indices

    @property
    def batch_size(self):
        return self._batch_size


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
    min_queue_examples = input_queue_memory_factor*EXAMPLES_PER_SHARD

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


def _setup_filename_queue(data_dir,
                          record_prefix,
                          num_readers,
                          should_shuffle,
                          capacity):
    """Sets up a filename queue of example-containing TFRecord files.
    """
    data_filenames = tf.gfile.Glob(
        os.path.join(data_dir, record_prefix + '*tfrecord'))
    assert data_filenames, ('No data files found.')
    assert len(data_filenames) >= num_readers

    filename_queue = tf.train.string_input_producer(
        string_tensor=data_filenames,
        shuffle=should_shuffle,
        capacity=capacity)

    return filename_queue


def _setup_batch_queue(images_and_joints, batch_size, num_preprocess_threads):
    """Sets up a batch queue that returns, e.g., a batch of 32 each of images,
    sparse joints and sparse joint indices.
    """
    images, joint_indices, joints = tf.train.batch_join(
        tensors_list=images_and_joints,
        batch_size=batch_size,
        capacity=2*num_preprocess_threads*batch_size)

    tf.summary.image(name='images', tensor=images)

    return TrainingBatch(images, joints, joint_indices, batch_size)


def setup_eval_input_pipeline(data_dir,
                              batch_size,
                              num_preprocess_threads,
                              image_dim):
    """Sets up an input pipeline for model evaluation.
    """
    filename_queue = _setup_filename_queue(data_dir, 'test', 1, False, 1)

    reader = tf.TFRecordReader()
    _, example_serialized = reader.read(filename_queue)

    # TODO(brendan): This should not call the same function as
    # `setup_train_input_pipeline`, as during training we will want to do image
    # distortion. But there is no image distortion yet, so this is fine for
    # now.
    images_and_joints = _parse_and_preprocess_images(
        example_serialized,
        num_preprocess_threads,
        image_dim)

    return _setup_batch_queue(images_and_joints, batch_size, num_preprocess_threads)


def setup_train_input_pipeline(data_dir,
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
        TrainingBatch(images, joints, joint_indices, batch_size): List of image
        tensors with first dimension (shape[0]) equal to batch_size, along with
        sparse vectors of joints (ground truth vectors), and sparse joint
        indices.
    """
    # TODO(brendan): num_readers == 1 case
    assert num_readers > 1

    with tf.name_scope('batch_processing'):
        filename_queue = _setup_filename_queue(
            data_dir, 'train', num_readers, True, 16)

        example_serialized = _setup_example_queue(filename_queue,
                                                  num_readers,
                                                  input_queue_memory_factor,
                                                  batch_size)

        images_and_joints = _parse_and_preprocess_images(
            example_serialized, num_preprocess_threads, image_dim)

        return _setup_batch_queue(images_and_joints,
                                  batch_size,
                                  num_preprocess_threads)
