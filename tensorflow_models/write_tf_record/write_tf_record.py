import threading
import numpy as np
import tensorflow as tf
from mpii_read import mpii_read
from timethis import timethis

tf.app.flags.DEFINE_integer('num_threads', 3,
                            """Number of threads to use to write TF Records""")

class ImageCoder(object):
    """A class that holds a session, passed using dependency injection during
    `ImageCoder` instantiations, which is used to run a TF graph to decode JPEG
    images.
    """
    def __init__(self, session):
        self._sess = session

        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data,
                                                 channels=3)
        shape = tf.shape(self._decode_jpeg)

        target_dim = tf.maximum(shape[0], shape[1])
        pad_image = tf.image.pad_to_bounding_box(self._decode_jpeg,
                                                 0,
                                                 0,
                                                 target_dim,
                                                 target_dim)

        resize_image = tf.image.resize_image_with_crop_or_pad(
            pad_image, 220, 220)

        self._scaled_image_jpeg = tf.image.encode_jpeg(resize_image)

    def decode_scale_encode(self, image_data):
        fetches = [self._decode_jpeg, self._scaled_image_jpeg]
        image, scaled_image_jpeg = self._sess.run(
            fetches, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3

        return image, scaled_image_jpeg


def _clamp01(value):
    """Clamps value to the range [0.0, 1.0].

    Args:
        value: A number to be clamped.

    Return:
        value if value is in [0.0, 1.0], otherwise whichever of 0.0 or 1.0 is
        closer to value.
    """
    return max(0, min(value, 1))


def _bytes_feature(value):
    """Wrapper for inserting bytes feature into Example proto"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Wrapper for inserting FloatList feature into Example proto"""
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Wrapper for inserting Int64 features into Example proto"""
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _append_scaled_joint(joints, joint_dim, max_joint_dim):
    """Appends to joints the value of joint_dim, scaled down to be in the range [0.0, 1.0].

    Args:
        joints: List of joints, in the order [x0, y0, x1, y1, ...]
        joint_dim: Next either xi or yi to append.
        max_joint_dim: Maximum dimension of the image in which the joint
            appears, e.g. height of 1080 in a 1920x1080 image.
    """
    scaled_joint = _clamp01(joint_dim/max_joint_dim)
    joints.append(scaled_joint)


def _extract_labeled_joints(people_in_img, image_shape):
    """Extracts all of the joints for all of the people in image, and puts them
    in a list in the format [x0, y0, x1, y1, ...].

    Not all joints are labeled for each person, so this function also returns a
    list of int64 bitmaps, one for each person, where the first 16 bits are 1
    if the joint is labeled, and 0 if the joint is not labeled.

    Args:
        people_in_img: List of people in this image.
        image_shape: The shape of the given image, in the format (y, x) or
            (rows, cols).

    Returns:
        (joints, joints_bitmaps) tuple, where both are lists. `joints` is a
        flat list with all of the joints in the image in sequence. The order
        can be determined from `joints_bitmaps`. Each int64 in `joints_bitmaps`
        corresponds to a person, so the joints for each person can be found by
        iterating over `joints` and extracting (x0, y0) pairs for each 1 bit in
        the `joints_bitmaps[i]` value for each person.

        Visually: `joints` [x0, y0, x1, y1, x2, y2]
                  `joints_bitmaps` [0b11, 0b10]

                  The above corresponds to two people, where (x0, y0) and
                  (x1, y1) are joints 0 and 1 for person 0, respectively, and
                  (x2, y2) is joint 1 for person 1.
    """
    joints = []
    joints_bitmaps = []
    for person in people_in_img:
        joint_bitmap = 0
        for joint_index in range(len(person.joints)):
            joint = person.joints[joint_index]
            if joint is not None:
                _append_scaled_joint(joints, joint[0], image_shape[1])
                _append_scaled_joint(joints, joint[1], image_shape[0])
                joint_bitmap = joint_bitmap | (1 << joint_index)

        joints_bitmaps.append(joint_bitmap)

    return joints, joints_bitmaps


def _write_example(coder, image_jpeg, people_in_img, writer):
    """Writes an example to the TFRecord file owned by `writer`.

    See `_extract_labeled_joints` for the format of `joints` and
    `joints_bitmaps`.
    """
    image, scaled_image_jpeg = coder.decode_scale_encode(image_jpeg)
    joints, joints_bitmaps = _extract_labeled_joints(people_in_img, image.shape)

    # TODO(brendan): height and width are known from JPEG format; don't encode
    # in TFRecord?
    # Rename `joints_bitmaps` to `joint_bitmaps`
    example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_jpeg': _bytes_feature(scaled_image_jpeg),
                    'joints_bitmaps': _int64_feature(joints_bitmaps),
                    'joints': _float_feature(joints)
                }))
    writer.write(example.SerializeToString())


def _process_image_files_single_thread(coder, thread_index, ranges, mpii_dataset):
    """Processes a range of filenames and labels in the MPII dataset
    corresponding to the given thread index.

    Args:
        coder: An instance of `ImageCoder`, which is used to decode JPEG images
            from MPII.
        thread_index: Index of the current thread (must be unique).
        mpii_dataset: Instance of `MpiiDataset` containing data shuffled in the
            order that those data should be written to TF Record.
    """
    tfrecord_filename = 'train{}.tfrecord'.format(thread_index)
    with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
        for img_index in range(ranges[thread_index][0], ranges[thread_index][1]):
            with tf.gfile.FastGFile(mpii_dataset.img_filenames[img_index], 'rb') as f:
                image_jpeg = f.read()

            _write_example(coder,
                           image_jpeg,
                           mpii_dataset.people_in_imgs[img_index],
                           writer)


def _process_image_files(mpii_dataset, num_examples, session):
    """Processes the image files in `mpii_dataset`, using multiple threads to
    write the data to TF Records on disk.
    """
    # TODO(brendan): Better documentation about `Coordinator`
    coord = tf.train.Coordinator()

    num_threads = tf.app.flags.FLAGS.num_threads

    spacing = np.linspace(0, num_examples, num_threads + 1).astype(np.int)
    ranges = []
    for spacing_index in range(len(spacing) - 1):
        ranges.append([spacing[spacing_index], spacing[spacing_index + 1]])

    coder = ImageCoder(session)

    threads = []
    for thread_index in range(num_threads):
        args = (coder, thread_index, ranges, mpii_dataset)
        t = threading.Thread(target=_process_image_files_single_thread, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)


@timethis
def write_tf_record(mpii_dataset, num_examples=None):
    # TODO(brendan): Docstring...
    with tf.Graph().as_default():
        with tf.Session() as session:
            if num_examples == None:
                num_examples = len(mpii_dataset.img_filenames)

            _process_image_files(mpii_dataset, num_examples, session)


def main(argv):
    """Usage:
    ('python3 -m write_tf_record
     "/mnt/data/datasets/MPII_HumanPose/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"
     --num_threads 3')
    """
    mpii_dataset = mpii_read(argv[1])
    write_tf_record(mpii_dataset)


if __name__ == "__main__":
    tf.app.run()
