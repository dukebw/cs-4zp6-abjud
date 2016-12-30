import threading
import numpy as np
import tensorflow as tf
from mpii_read import mpii_read
from timethis import timethis

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_threads', 4,
                            """Number of threads to use to write TF Records""")
tf.app.flags.DEFINE_integer('image_dim', 299,
                            """Dimension of the square image to output.""")

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rectangle(object):
    def __init__(self, rect):
        self.top_left = Point(rect[0], rect[1])
        self.bottom_right = Point(rect[2], rect[3])

    def get_height(self):
        return self.bottom_right.y - self.top_left.y

    def get_width(self):
        return self.bottom_right.x - self.top_left.x


class ImageCoder(object):
    """A class that holds a session, passed using dependency injection during
    `ImageCoder` instantiations, which is used to run a TF graph to decode JPEG
    images.

    On initialization, a graph is set up containing the following operations:
        1. Decode an input JPEG image.
        2. Crop the raw image to an input bounding box, e.g. the box around a
           person.
        3. Pad the shorter dimension with a black border, to a square size.
        4. Resize the now square image to FLAGS.image_dim*FLAGS.image_dim.
        5. Encode the cropped, resized image as JPEG.
    """
    def __init__(self, session):
        self._sess = session

        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        raw_image = tf.image.decode_jpeg(contents=self._decode_jpeg_data, channels=3)
        self._img_shape = tf.shape(input=raw_image)

        self._crop_height_offset = tf.placeholder(dtype=tf.int32)
        self._crop_width_offset = tf.placeholder(dtype=tf.int32)
        self._crop_height = tf.placeholder(dtype=tf.int32)
        self._crop_width = tf.placeholder(dtype=tf.int32)
        self._height_pad = tf.placeholder(dtype=tf.int32)
        self._width_pad = tf.placeholder(dtype=tf.int32)
        self._padded_img_dim = tf.placeholder(dtype=tf.int32)

        cropped_img = tf.image.crop_to_bounding_box(image=raw_image,
                                                    offset_height=self._crop_height_offset,
                                                    offset_width=self._crop_width_offset,
                                                    target_height=self._crop_height,
                                                    target_width=self._crop_width)

        pad_image = tf.image.pad_to_bounding_box(image=cropped_img,
                                                 offset_height=self._height_pad,
                                                 offset_width=self._width_pad,
                                                 target_height=self._padded_img_dim,
                                                 target_width=self._padded_img_dim)

        self._scaled_image_tensor = tf.cast(
            tf.image.resize_images(images=pad_image, size=[FLAGS.image_dim, FLAGS.image_dim]),
            tf.uint8)

        self._scaled_image_jpeg = tf.image.encode_jpeg(image=self._scaled_image_tensor)

    def decode_jpeg(self, image_data):
        """Returns the shape of an input JPEG image.

        Args:
            image_data: A JPEG image to find the shape of.

        Returns:
            shape: Shape of the image in the format Point(width, height).
        """
        shape = self._sess.run(fetches=self._img_shape,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(shape) == 3
        assert shape[2] == 3

        return Point(shape[1], shape[0])

    def scale_encode(self,
                     image_data,
                     crop_offsets,
                     crop_dim,
                     padding,
                     padded_dim):
        """Runs the entire sequence of decode -> crop -> pad -> resize ->
        encode JPEG, and returns the resultant JPEG image.

        Args:
            image_data: JPEG image data.
            crop_offsets: A `Point` containing the offset in the original image
                of the sub-image to crop to.
            crop_dim: A `Point` containing the width and height of the cropped
                section, in the format Point(width, height).
            padding: Amount of padding to do on the cropped image in the format
                Point(x_padding, y_padding).
            padded_dim: Length of the edge length of the square padded image.

        Returns: The image cropped and padded to the given bounding box, scaled
            to FLAGS.image_dim*FLAGS.image_dim, and encoded as JPEG.
        """
        feed_dict = {
            self._decode_jpeg_data: image_data,
            self._crop_height_offset: crop_offsets.y,
            self._crop_width_offset: crop_offsets.x,
            self._crop_height: crop_dim.y,
            self._crop_width: crop_dim.x,
            self._height_pad: padding.y,
            self._width_pad: padding.x,
            self._padded_img_dim: padded_dim
        }

        scaled_img_jpeg = self._sess.run(fetches=self._scaled_image_jpeg,
                                         feed_dict=feed_dict)

        return scaled_img_jpeg


def _clamp_range(value, min_val, max_val):
    """Clamps value to the range [min_val, max_val].

    Args:
        value: A number to be clamped.
        min_val: Minimum value to return.
        max_val: Maximum value to return.

    Return:
        value if value is in [min_val, max_val], otherwise whichever of
        `min_val` or `max_val` is closer to value.
    """
    return max(min_val, min(value, max_val))


def _clamp_point_to_image(point, image_shape):
    """Clamps `point` so that it is inside the image whose shape is given by
    `image_shape`.

    Args:
        point: `Point` to clamp.
        image_shape: Dimensions of an image in the format Point(width, height).

    Returns:
        clamped_point: `point` with its dimensions clamped to the edges of the
            image.
    """
    clamped_point = (_clamp_range(point.x, 0, image_shape.x),
                     _clamp_range(point.y, 0, image_shape.y))

    return clamped_point


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


def _append_scaled_joint(joints, joint_dim, max_joint_dim, image_center):
    """Appends to joints the value of joint_dim, scaled down to be in the range
    [-1.0, 1.0].

    Args:
        joints: List of joints, in the order [x0, y0, x1, y1, ...]
        joint_dim: Next either xi or yi to append.
        max_joint_dim: Maximum dimension of the image in which the joint
            appears, e.g. height of 1080 in a 1920x1080 image.
    """
    scaled_joint = _clamp_range(
        (joint_dim - image_center)/max_joint_dim, -1, 1)
    joints.append(scaled_joint)


def _extract_labeled_joints(person,
                            image_shape,
                            padding,
                            offsets):
    """Extracts all of the joints for a single person in image, and puts them
    in a list in the format [x0, y0, x1, y1, ...].

    Not all joints are labeled for each person, so this function also returns a
    list of int64 bitmaps for person, where the first 16 bits are 1 if the
    joint is labeled, and 0 if the joint is not labeled.

    Args:
        person: Person in the image to get joints for.
        image_shape: The shape of the given image, in the format
            Point(cols, rows).
        padding: Pixels of padding in Point(width, height) dimensions.
        offsets: The Point(width, height) offsets of this cropped image in the
            original image. This is needed to translate the joint labels.

    Returns:
        (joints, joint_bitmap) tuple, where `joints` is a list, and
        joint_bitmap is an integer. `joints` is a flat list with all of the
        joints in the image in sequence. The order can be determined from
        `joint_bitmap`. The joints for `person` can be found by extracting (x0,
        y0) pairs for each 1 bit in `joint_bitmap`.

        Visually: `joints` [x0, y0, x1, y1, x2, y2]
                  `joint_bitmap` 0b1011

                  The above corresponds to a person for whom (x0, y0), (x1, y1)
                  and (x2, y2) are joints 0, 1 and 3 for `person`,
                  respectively.
    """
    joints = []
    max_image_dim = max(image_shape.x, image_shape.y)
    image_center = int(max_image_dim/2)

    joint_bitmap = 0
    for joint_index in range(len(person.joints)):
        joint = person.joints[joint_index]
        if joint is not None:
            joint = Point(joint[0], joint[1])
            if ((offsets.x <= joint.x <= (offsets.x + image_shape.x)) and
                (offsets.y <= joint.y <= (offsets.y + image_shape.y))):
                joint.x -= offsets.x
                joint.y -= offsets.y
                _append_scaled_joint(joints,
                                     joint.x + padding.x,
                                     max_image_dim,
                                     image_center)
                _append_scaled_joint(joints,
                                     joint.y + padding.y,
                                     max_image_dim,
                                     image_center)
                joint_bitmap |= (1 << joint_index)

    return joints, joint_bitmap


def _find_person_bounding_box(person, img_shape):
    """Finds an enclosing bounding box for `person` in the image with
    `img_shape` dimensions.

    Currently the bounding box is found by taking a generous multiple of the
    size of the person's head bounding box, which is encoded in the `Person`
    object.

    This has the issue of potentially cropping people in weird positions, for
    example upside-down people.

    One improvement would be to take the bounding box found with the current
    method, and expand each dimension so that all the labelled joints are
    contained.

    Args:
        person: Person to find bounding box for.
        img_shape: Dimensions of the image that the person is in.

    Returns:
        A `Rectangle` describing the box bounding `person`.
    """
    head_rect = Rectangle(person.head_rect)
    head_width = head_rect.get_width()
    head_height = head_rect.get_height()

    top_left = Point(head_rect.top_left.x - 4*head_width,
                     head_rect.top_left.y - head_height)
    bottom_right = Point(head_rect.top_left.x + 4*head_width,
                         head_rect.top_left.y + 7*head_height)

    return Rectangle(_clamp_point_to_image(top_left, img_shape) +
                     _clamp_point_to_image(bottom_right, img_shape))

def _find_padded_person_dim(person_rect):
    """Finds the large dimension, shape and padding needed for the bounding box
    around a person.

    Args:
        person_rect: A `Rectangle` describing the bounding box of a person.

    Returns:
        padded_img_dim: Larger dimension of the person's bounding box.
        person_shape_xy: Point(width, height) describing the person's
            bounding box dimensions.
        padding_xy: Point(padding_x, padding_y) describing the padding for the
            person in the x and y dimensions, at least one of which will be
            zero.
    """
    person_width = person_rect.get_width()
    person_height = person_rect.get_height()
    padding = int(abs(person_height - person_width)/2)
    if person_height > person_width:
        height_pad = 0
        width_pad = padding
    else:
        height_pad = padding
        width_pad = 0

    padded_img_dim = max(person_width, person_height)
    person_shape_xy = Point(person_width, person_height)
    padding_xy = Point(width_pad, height_pad)

    return padded_img_dim, person_shape_xy, padding_xy

def _write_example(coder, image_jpeg, people_in_img, writer):
    """Writes an example to the TFRecord file owned by `writer`.

    See `_extract_labeled_joints` for the format of `joints` and
    `joint_bitmap`.
    """
    img_shape = coder.decode_jpeg(image_jpeg)

    for person in people_in_img:
        person_rect = _find_person_bounding_box(person, img_shape)

        padded_img_dim, person_shape_xy, padding_xy = _find_padded_person_dim(
            person_rect)

        scaled_img_jpeg = coder.scale_encode(
            image_jpeg,
            person_rect.top_left,
            person_shape_xy,
            padding_xy,
            padded_img_dim)

        joints, joint_bitmap = _extract_labeled_joints(person,
                                                       person_shape_xy,
                                                       padding_xy,
                                                       person_rect.top_left)

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_jpeg': _bytes_feature(scaled_img_jpeg),
                    'joint_bitmap': _int64_feature(joint_bitmap),
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
    with tf.python_io.TFRecordWriter(path=tfrecord_filename) as writer:
        for img_index in range(ranges[thread_index][0], ranges[thread_index][1]):
            with tf.gfile.FastGFile(name=mpii_dataset.img_filenames[img_index], mode='rb') as f:
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

    num_threads = FLAGS.num_threads

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
