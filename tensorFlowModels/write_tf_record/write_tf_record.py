import threading
from PIL import Image, ImageDraw
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
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3

        return image


def _clamp01(value):
    """Clamps value to the range [0.0, 1.0].

    Args:
        value: A number to be clamped.

    Return:
        value if value is in [0.0, 1.0], otherwise whichever of 0.0 or 1.0 is
        closer to value.
    """
    return max(0, min(value, 1))


def _show_image_with_joints(next_image, people_in_img):
    """Debug function to display an image with the head rectangle and joints.

    Args:
        next_image: The image to display.
        people_in_img: the list of `Person`s corresponding to next_image from
            the MpiiDataset class.
    """
    image = Image.fromarray(next_image)
    draw = ImageDraw.Draw(image)

    for person in people_in_img:
        draw.rectangle(person.head_rect, outline=(0, 0xFF, 0))

        for joint_index in range(len(person.joints)):
            joint = person.joints[joint_index]

            if joint:
                red = int(0xFF*(joint_index % 5)/5)
                green = int(0xFF*(joint_index % 10)/10)
                blue = int(0xFF*joint_index/16)
                colour = (red, green, blue)
                box = (joint[0] - 5, joint[1] - 5,
                       joint[0] + 5, joint[1] + 5)
                draw.ellipse(box, fill=colour)

    image.show()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _write_example(image, writer):
    """Writes an example.
    """
    # TODO(brendan): Docstring
    image_raw = image.tostring()

    example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_raw': _bytes_feature(image_raw),
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

            image = coder.decode_jpeg(image_jpeg)
            _write_example(image, writer)
            

def _process_image_files_with_filename_queue(mpii_dataset,
                                             num_examples,
                                             session):
    """Creates a TF Queue (queue of Tensors) of filenames, reads and decodes
    them by running the default graph, then writes the example to a TF Record.

    Currently unused.
    """
    # TODO(brendan): The performance of this method is more than 2x slower than
    # running multiple Python threads (the `_process_image_files` method). See
    # README.md.
    # It may be an interesting experiment to see if the entire TFRecord writing
    # process can be moved to TensorFlow, and whether that is faster.
    filename_queue = tf.train.string_input_producer(mpii_dataset.img_filenames,
                                                    shuffle=False)
    _, img_jpeg = tf.WholeFileReader().read(filename_queue)
    img_tensor = tf.image.decode_jpeg(img_jpeg)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    with tf.python_io.TFRecordWriter('train.tfrecord') as writer:
        for img_index in range(num_examples):
            next_image = session.run(img_tensor)
            _write_example(next_image, writer)

    coord.request_stop()
    coord.join(threads)

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
