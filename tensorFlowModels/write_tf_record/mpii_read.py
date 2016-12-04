"""
A module for reading the [MPII Human Pose
Dataset](http://human-pose.mpi-inf.mpg.de/).

MPII Dataset

The dataset is a set of 25 000 images, containing annotations for 40 000 people
in total. The images are colour and of various dimensions (e.g. 1280x720,
640x480 etc.).

All of the images, along with corresponding data, are stored in a Matlab
structure called `RELEASE`.

The `scale` and `objpos` fields of each annotation field `annorect` of
`annolist` can be used to find the scale of the human in the image (with
respect to a height of 200px), and the rough human position in the image.

Only training images, i.e. images with field `img_train` of structure `RELEASE`
equal to 1, have corresponding labels. Joint labels can be found from the
(TODO(brendan): point labels explanation). Coordinates of the head rectangle
can be found from the `x1`, `y1`, `x2`, and `y2` fields of `annorect`.
"""
import sys
import os
import scipy.io
import tensorflow as tf
import PIL.Image
import numpy as np

class Person(object):
    """
    The joints should be a list of (x, y) tuples where x and y are both in the
    range [0.0, 1.0], and the joint ids are as follows,

    0 - r ankle
    1 - r knee
    2 - r hip
    3 - l hip
    4 - l knee
    5 - l ankle
    6 - pelvis
    7 - thorax
    8 - upper neck
    9 - head top
    10 - r wrist
    11 - r elbow
    12 - r shoulder
    13 - l shoulder
    14 - l elbow
    15 - l wrist
    """
    NUM_JOINTS = 16

    def __init__(self, joints, head_rect):
        self._joints = Person.NUM_JOINTS*[None]
        for joint in joints:
            self._joints[joint.id] = (joint.x, joint.y)

        self._headrect = head_rect

    @property
    def joints(self):
        return self._joints

    @property
    def head_rect(self):
        return self._head_rect

class MpiiDataset(object):
    """
    Representation of the entire MPII Dataset.

    The annotation description can be found
    [here](http://human-pose.mpi-inf.mpg.de/#download).

    Currently only the images and person-centric body joint annotations are
    taken from the dataset.
    """
    def __init__(self, images, people):
        self._images = images
        self._people = people

    @property
    def images(self):
        return self._images

    @property
    def people(self):
        return self._people

def mpii_read(mpii_dataset_filepath):
    """
    Note that the images are assumed to reside in a folder one up from the .mat
    file that is being parsed (i.e. ../images).
    """
    mpii_dataset_mat = scipy.io.loadmat(mpii_dataset_filepath,
                                        struct_as_record = False,
                                        squeeze_me = True)['RELEASE']
    mpii_annotations = mpii_dataset_mat.annolist
    train_or_test = mpii_dataset_mat.img_train

    img_filenames = []
    people = []
    for img_index in range(len(mpii_annotations)):
        if train_or_test[img_index] == 1:
            img_annotation = mpii_annotations[img_index]

            mpii_dataset_dir = os.path.dirname(mpii_dataset_filepath)
            img_abs_filepath = os.path.join(mpii_dataset_dir,
                                            '../images',
                                            img_annotation.image.name)
            img_filenames.append(img_abs_filepath)

            if not hasattr(img_annotation.annorect, '__iter__'):
                img_annotation.annorect = [img_annotation.annorect]

            for img_annorect in img_annotation.annorect:
                head_rect = (img_annorect.x1, img_annorect.y1,
                             img_annorect.x2, img_annorect.y2)

                people.append(Person(img_annorect.annopoints.point, head_rect))

    # TODO(brendan): augment? draw?

    return img_filenames, people

if __name__ == "__main__":
    assert len(sys.argv) == 1

    img_filenames, people = mpii_read(sys.argv[0])

    _, img_jpeg = tf.WholeFileReader().read(img_filenames)
    image = tf.image.decode_jpeg(img_jpeg)

    sess = tf.InteractiveSession()
