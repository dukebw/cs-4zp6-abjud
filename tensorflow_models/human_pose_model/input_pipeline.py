import os
import numpy as np
import tensorflow as tf
from sparse_to_dense import sparse_joints_to_dense_single_example
from mpii_read import Person

EXAMPLES_PER_SHARD = 1024

NUM_JOINTS = Person.NUM_JOINTS

class EvalBatch(object):
    """Contains an evaluation batch of images along with corresponding
    ground-truth joint vectors for the annotated person in that image.

    images, *_joints, joint_indices and head_size should all be lists of length
    `batch_size`.
    """
    def __init__(self,
                 images,
                 binary_maps,
                 heatmaps,
                 x_joints,
                 y_joints,
                 joint_indices,
                 head_size,
                 batch_size):

        assert images.get_shape()[0] == batch_size
        self._images = images
        self._binary_maps = binary_maps
        self._heatmaps = heatmaps
        self._x_joints = x_joints
        self._y_joints = y_joints
        self._joint_indices = joint_indices
        self._head_size = head_size
        self._batch_size = batch_size

    @property
    def images(self):
        return self._images

    @property
    def binary_maps(self):
        return self._binary_maps

    @property
    def heatmaps(self):
        return self._heatmaps

    @property
    def x_joints(self):
        return self._x_joints

    @property
    def y_joints(self):
        return self._y_joints

    @property
    def joint_indices(self):
        return self._joint_indices

    @property
    def head_size(self):
        return self._head_size

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


def _parse_example_proto(example_serialized, image_dim):
    """Parses an example proto and returns a tuple containing
    (raw image reshaped to the image dimensions in float32 format, sparse joint
    indices, sparse joints).
    """
    feature_map = {
        'image_jpeg': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'binary_maps': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'heatmaps': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'joint_indices': tf.VarLenFeature(dtype=tf.int64),
        'x_joints': tf.VarLenFeature(dtype=tf.float32),
        'y_joints': tf.VarLenFeature(dtype=tf.float32),
        'head_size': tf.FixedLenFeature(shape=[], dtype=tf.float32)
    }
    features = tf.parse_single_example(
        serialized=example_serialized, features=feature_map)

    img_jpeg = features['image_jpeg']
    with tf.name_scope(name='decode_jpeg', values=[img_jpeg]):
        img_tensor = tf.image.decode_jpeg(contents=img_jpeg,
                                          channels=3)
        decoded_img = tf.image.convert_image_dtype(
            image=img_tensor, dtype=tf.float32)

    parsed_example = (decoded_img,
                      features['binary_maps']
                      features['heatmaps']
                      features['joint_indices'],
                      features['x_joints'],
                      features['y_joints'],
                      features['head_size'])

    return parsed_example

# DEPRECATED 08FEB17
def _get_renormalized_joints(joints, old_img_dim, new_center, new_img_dim):
    """Renormalizes a 1-D vector of joints to a new co-ordinate system given by
    `new_center`, and returns the `SparseTensor` of joints, renormalized.

    N(x, b') = 1/w'*(x - x_c')
             = 1/w'*(w*N(x; b) + (x_c - x_c'))
    """
    new_joint_vals = (1/new_img_dim*
                      (old_img_dim*tf.cast(joints.values, tf.float64) + (old_img_dim/2 - new_center)))

    return tf.SparseTensor(joints.indices, new_joint_vals, joints.shape)

def _distort_colour(distorted_image, thread_id):
    """Distorts the brightness, saturation, hue and contrast of an image
    randomly, and returns the result.

    The colour distortions are non-commutative, so we do them in a random order
    per thread (based on `thread_id`).
    """
    colour_ordering = thread_id % 2

    distorted_image = tf.image.random_brightness(image=distorted_image, max_delta=32./255.)
    if colour_ordering == 0:
        distorted_image = tf.image.random_saturation(image=distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_hue(image=distorted_image, max_delta=0.2)
        distorted_image = tf.image.random_contrast(image=distorted_image, lower=0.5, upper=1.5)
    else:
        distorted_image = tf.image.random_contrast(image=distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_saturation(image=distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_hue(image=distorted_image, max_delta=0.2)

    return tf.clip_by_value(t=distorted_image, clip_value_min=0.0, clip_value_max=1.0)


def _randomly_crop_image(decoded_img,
                         x_joints,
                         y_joints,
                         image_dim,
                         thread_id):
    """Randomly crops `deocded_img` to a bounding box covering at least 0.9 of
    the image, and maintaining an aspect ratio between 0.75 and 1.33.

    Joints have to be re-normalized such that their values have been translated
    into the new cropped image space. The normalized joint co-ordinates are the
    distance from the image center as a proportion of the dimension of the
    image. The renormalization equation can be found in the
    `_get_renormalized_joints` docstring.

    Args:
        decoded_img: Decoded image produced by parsing a serialized example and
            decoding its JPEG image.
        x_joints: `SparseTensor` of x coordinates of joints returned by
            de-serializing an example.
        y_joints: `SparseTensor` of y coordinates of joints returned by
            de-serializing an example.
        image_dim: Dimension of the image as required when input to the
            network.
        thread_id: Number of the image preprocessing thread responsible for
            these image distortions.

    Returns:
        Tuple (distorted_image, x_joints, y_joints) resulting from randomly
        cropping the image and renormalizing the joints.
    """
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        image_size=[image_dim, image_dim, 3],
        bounding_boxes=[[[0, 0, 1.0, 1.0]]],
        min_object_covered=1.0,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.5, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)

    distorted_image = tf.slice(input_=decoded_img,
                               begin=bbox_begin,
                               size=bbox_size)

    distorted_binary_maps = tf.slice(input_=binary_maps,
                                    begin=bbox_begin,
                                    size=bbox_size)

    distorted_heatmaps = tf.slice(input_=heatmaps,
                                    begin=bbox_begin,
                                    size=bbox_size)

    distorted_image = tf.image.resize_images(images=[distorted_image],
                                             size=[image_dim, image_dim],
                                             method=thread_id % 4,
                                             align_corners=False)

    distorted_binary_maps = tf.image.resize_images(images=[distorted_binary_maps],
                                             size=[image_dim, image_dim],
                                             method=thread_id % 4,
                                             align_corners=False)

    distorted_heatmaps = tf.image.resize_images(images=[distorted_heatmaps],
                                             size=[image_dim, image_dim],
                                             method=thread_id % 4,
                                             align_corners=False)

    distorted_image = tf.reshape(
        tensor=distorted_image,
        shape=[image_dim, image_dim, 3])

    distorted_binary_maps = tf.reshape(
        tensor=distorted_binary_maps,
        shape=[image_dim, image_dim, 16])

    distorted_heatmaps = tf.reshape(
        tensor=distorted_heatmaps,
        shape=[image_dim, image_dim, 16])

    """ DEPRECATED 08FEB17
    distorted_center = tf.cast(bbox_begin, tf.float64) + bbox_size/2

    distorted_center_y = distorted_center[0]
    distorted_center_x = distorted_center[1]
    distorted_height = bbox_size[0]
    distorted_width = bbox_size[1]

    x_joints = _get_renormalized_joints(x_joints,
                                        image_dim,
                                        distorted_center_x,
                                        distorted_width)
    y_joints = _get_renormalized_joints(y_joints,
                                        image_dim,
                                        distorted_center_y,
                                        distorted_height)
    """
    return distorted_image, x_joints, y_joints


def _distort_image(parsed_example, image_dim, thread_id):
    """Randomly distorts the image from `parsed_example` by randomly cropping,
    randomly flipping left and right, and randomly distorting the colour of
    that image.

    Args:
        parsed_example: Tuple (decoded_img, joint_indices, x_joints, y_joints, head_size)
            returned from parsing a serialized example protobuf.
        image_dim: Dimension of the image as required when input to the
            network.
        thread_id: Number of the image preprocessing thread responsible for
            these image distortions.

    Returns:
        (distorted_image, joint_indices, x_joints, y_joints) tuple containing
        all information from `parsed_example`, except the image has been
        distorted and the joints have been renormalized to account for those
        distortions.
    """
    decoded_image, binary_maps, heatmaps, joint_indices, x_joints, y_joints, _ = parsed_example

    distorted_image, distorted_binary_maps, distorted_heatmaps  = _randomly_crop_image(decoded_image,
                                                                                       binary_maps,
                                                                                       heatmaps,
                                                                                       x_joints,
                                                                                       y_joints,
                                                                                       image_dim,
                                                                                       thread_id)

    rand_uniform = tf.random_uniform(shape=[],
                                     minval=0,
                                     maxval=1.0)
    should_flip = rand_uniform < 0.5
    distorted_image = tf.cond(
        pred=should_flip,
        fn1=lambda: tf.image.flip_left_right(image=distorted_image),
        fn2=lambda: distorted_image)

    distorted_binary_maps = tf.cond(
        pred=should_flip,
        fn1=lambda: tf.image.flip_left_right(image=distorted_binary_maps),
        fn2=lambda: distorted_image)

    distorted_heatmaps = tf.cond(
        pred=should_flip,
        fn1=lambda: tf.image.flip_left_right(image=distorted_heatmaps),
        fn2=lambda: distorted_image)

    distorted_image = _distort_colour(distorted_image, thread_id)

    return distorted_image, distorted_binary_maps, distorted_heatmaps, joint_indices, x_joints, y_joints


def _get_joints_normal_pdf(dense_joints, std_dev, coords, expand_axis):
    """Creates a set of 1-D Normal distributions with means equal to the
    elements of `dense_joints`, and the standard deviations equal to the values
    in `std_dev`.

    The shapes of the input tensors `dense_joints` and `std_dev` must be the
    same.

    Args:
        dense_joints: Dense tensor of ground truth joint locations in 1-D.
        std_dev: Standard deviation to use for the normal distribution.
        coords: Set of co-ordinates over which to evaluate the Normal's PDF.
        expand_axis: Axis to expand the probabilities by using
            `tf.expand_dims`. E.g., an `expand_axis` of -2 will turn the result
            PDF output from a tensor with shape [16, 380] to a tensor with
            shape [16, 380, 1]

    Returns:
        3-D tensor with first dim being the length of `dense_joints`, and two
        more dimensions, one of which is the length of `coords` and the other
        of which has size 1.
    """
    normal = tf.contrib.distributions.Normal(mu=dense_joints, sigma=std_dev)
    probs = normal.pdf(coords)
    probs = tf.transpose(probs)

    return tf.expand_dims(input=probs, axis=expand_axis)


def _parse_and_preprocess_example_eval(example_serialized,
                                       num_preprocess_threads,
                                       image_dim,
                                       is_train):
    """Same as `_parse_and_preprocess_example_train`, except without image
    distortion or heatmap creation.

    Hence this function returns dense x and y joints, instead of dense
    heatmaps.
    """
    images_and_joints = []
    for thread_id in range(num_preprocess_threads):
        parsed_example = _parse_example_proto(example_serialized, image_dim)

        decoded_img, binary_maps, heatmaps, joint_indices, x_joints, y_joints, head_size = parsed_example

        decoded_img = tf.reshape(
            tensor=decoded_img,
            shape=[image_dim, image_dim, 3])

        decoded_img = tf.sub(x=decoded_img, y=0.5)
        decoded_img = tf.mul(x=decoded_img, y=2.0)

        images_and_joints.append([decoded_img,
                                  binary_maps,
                                  heatmaps,
                                  x_joints,
                                  y_joints,
                                  joint_indices,
                                  head_size])

    return images_and_joints

def _parse_and_preprocess_example_train(example_serialized,
                                        num_preprocess_threads,
                                        image_dim,
                                        heatmap_stddev_pixels):
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
        heatmap_stddev_pixels: Standard deviation of Gaussian joint heatmap, in
            pixels.

    Returns:
        A list of lists, one for each thread, where each inner list contains a
        decoded image with colours scaled to range [-1, 1], as well as the
        sparse joint ground truth vectors.
    """
    images_and_joint_maps = []
    for thread_id in range(num_preprocess_threads):
        parsed_example = _parse_example_proto(example_serialized, image_dim)

        distorted_image, distorted_heatmaps, distorted_binary_maps = _distort_image(
            parsed_example, image_dim, thread_id)

        x_dense_joints, y_dense_joints, weights = sparse_joints_to_dense_single_example(
            x_joints, y_joints, joint_indices, NUM_JOINTS)

        distorted_image = tf.sub(x=distorted_image, y=0.5)
        distorted_image = tf.mul(x=distorted_image, y=2.0)

        images_and_joint_maps.append([distorted_image, distorted_heatmaps, distorted_binary_maps, weights])

    return images_and_joint_maps


def _setup_batch_queue(images_and_joint_maps, batch_size, num_preprocess_threads):
    """Sets up a batch queue that returns, e.g., a batch of 32 each of images,
    sparse joints and sparse joint indices.
    """
    images, binary_maps, heatmaps, weights = tf.train.batch_join(
        tensors_list=images_and_joint_maps,
        batch_size=batch_size,
        capacity=2*num_preprocess_threads*batch_size)

    image_dim =  heatmaps.get_shape().as_list()[1]

    merged_heatmaps = tf.reshape(tf.reduce_max(heatmaps,3),[batch_size,image_dim, image_dim, 1])
    merged_heatmaps = tf.cast(merged_heatmaps, tf.float32)

    merged_binary_maps = tf.reshape(tf.reduce_max(binary_maps,3),[batch_size,image_dim, image_dim, 1])
    merged_binary_maps = tf.cast(merged_binary_maps, tf.float32)

    tf.summary.image(name='images', tensor=images)
    tf.summary.image(name='heatmaps', tensor=merged_heatmaps)
    tf.summary.image(name='binary_maps', tensor=merged_binary_maps)

    return images, binary_maps, heatmaps, weights


def setup_eval_input_pipeline(batch_size,
                              num_preprocess_threads,
                              image_dim,
                              data_filenames):
    """Sets up an input pipeline for model evaluation.

    This function is similar to `setup_train_input_pipeline`, except that
    images are not distorted, and the filename queue will process only one
    TFRecord at a time. Therefore no Example queue is needed.
    """
    filename_queue = tf.train.string_input_producer(
        string_tensor=data_filenames,
        shuffle=False,
        capacity=1)

    reader = tf.TFRecordReader()
    _, example_serialized = reader.read(filename_queue)

    images_and_joint_maps = _parse_and_preprocess_example_eval(
        example_serialized,
        num_preprocess_threads,
        image_dim,
        False)

    #TODO
    images,binary_maps, heatmaps, x_joints, y_joints, joint_indices, head_size = tf.train.batch_join(
        tensors_list=images_and_joint_maps,
        batch_size=batch_size,
        capacity=2*num_preprocess_threads*batch_size)

    return EvalBatch(images,
                     binary_maps,
                     heatmaps,
                     x_joints,
                     y_joints,
                     joint_indices,
                     head_size,
                     batch_size)


def setup_train_input_pipeline(FLAGS, data_filenames):
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
        num_readers: Number of file readers to use.
        input_queue_memory_factor: Input to `_setup_example_queue`. See
            `_setup_example_queue` for details.
        batch_size: Number of examples to process at once (in one training
            step).
        num_preprocess_threads: Number of threads to use to preprocess image
            data.
        image_dim: Dimension of square input images.
        data_filenames: Set of filenames to get examples from.

    Returns:
        (images, heatmaps, weights, batch_size): List of image
        tensors with first dimension (shape[0]) equal to batch_size, along with
        lists of dense vectors of heatmaps of ground truth vectors (joints),
        dense vectors of weights and the batch size.
    """

    num_readers = FLAGS.num_readers
    input_queue_memory_factor = FLAGS.input_queue_memory_factor
    batch_size = FLAGS.batch_size
    num_preprocess_threads = FLAGS.num_preprocess_threads
    image_dim = FLAGS.image_dim

    # TODO(brendan): num_readers == 1 case
    assert num_readers > 1

    with tf.name_scope('batch_processing'):
        filename_queue = tf.train.string_input_producer(
            string_tensor=data_filenames,
            shuffle=True,
            capacity=16)

        example_serialized = _setup_example_queue(filename_queue,
                                                  num_readers,
                                                  input_queue_memory_factor,
                                                  batch_size)

        images_and_joint_maps = _parse_and_preprocess_example_train(
            example_serialized,
            num_preprocess_threads,
            image_dim)

        return _setup_batch_queue(images_and_joint_maps,
                                  batch_size,
                                  num_preprocess_threads)
