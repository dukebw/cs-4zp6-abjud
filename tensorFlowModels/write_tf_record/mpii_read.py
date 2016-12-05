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
annolist[index].annorect.annopoints.point field. Coordinates of the head
rectangle can be found from the `x1`, `y1`, `x2`, and `y2` fields of
`annorect`.

Out of the total dataset of 24987 images, 18079 of those are training images.
Of those images marked as training, 233 actually have no joint annotations, and
only have a head rectangle.
"""
import sys
import os
import scipy.io
import numpy as np

class Person(object):
    """A class representing each person in a given image, including their head
    rectangle and joints.

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

    Attributes:
        joints: A list of 16 joints for the person, which all default to
            `None`. Their values are potentially filled in from the MPII
            dataset annotations.
        head_rect: A tuple of four values (x1, y1, x2, y2) defining a rectangle
            around the head of the person in the image.
    """
    NUM_JOINTS = 16

    def __init__(self, joints, head_rect):
        self._joints = Person.NUM_JOINTS*[None]

        joints = _make_iterable(joints)

        for joint in joints:
            self._joints[joint.id] = (joint.x, joint.y)

        self._head_rect = head_rect

    @property
    def joints(self):
        return self._joints

    @property
    def head_rect(self):
        return self._head_rect

class MpiiDataset(object):
    """Representation of the entire MPII dataset.

    The annotation description can be found
    [here](http://human-pose.mpi-inf.mpg.de/#download).

    Currently only the images and person-centric body joint annotations are
    taken from the dataset.

    Attributes:
        img_filenames: A list of the names of the paths of each image.
        people_in_imgs: A list of lists of the `Person` class, where each list
            of `Person`s represents all the people in the image at the same
            list index. Must be the same length as `img_filenames`.
    """
    def __init__(self, img_filenames, people_in_imgs):
        self._img_filenames = img_filenames
        self._people_in_imgs = people_in_imgs

    @property
    def img_filenames(self):
        return self._img_filenames

    @property
    def people_in_imgs(self):
        return self._people_in_imgs

def _make_iterable(maybe_iterable):
    """Checks whether `maybe_iterable` is iterable, and if not returns an
    iterable structure containing `maybe_iterable`.

    Args:
        maybe_iterable: An object that may or may not be iterable.

    Returns:
        maybe_iterable: If `maybe_iterable` was iterable, then it is returned,
        otherwise an iterable structure containing `maybe_iterable` is
        returned.
    """
    if not hasattr(maybe_iterable, '__iter__'):
        maybe_iterable = [maybe_iterable]

    return maybe_iterable

def _parse_annotation(img_annotation, mpii_images_dir):
    """Parses a single image annotation from the MPII dataset.

    Looks at the annotations for a single image, and returns the people in the
    image along with the full filepath of the image.

    Args:
        img_annotation: The annotations coming from annolist(index) from the
            MPII dataset.
        mpii_images_dir: Path to the directory where the MPII images are.

    Returns:
        img_abs_filepath: Filepath of the image corresponding to
            `img_annotation`.
        people: A list of `Person`s corresponding to the annotated people in
            the image.
    """
    img_abs_filepath = os.path.join(mpii_images_dir,
                                    img_annotation.image.name)

    img_annotation.annorect = _make_iterable(img_annotation.annorect)

    people = []
    for img_annorect in img_annotation.annorect:
        head_rect = (img_annorect.x1, img_annorect.y1,
                     img_annorect.x2, img_annorect.y2)

        try:
            people.append(Person(img_annorect.annopoints.point, head_rect))
        except AttributeError:
            people.append(Person([], head_rect))

    return img_abs_filepath, people

def parse_mpii_data_from_mat(mpii_dataset_mat, mpii_images_dir):
    """Parses the training data out of `mpii_dataset_mat` into a `MpiiDataset`
    Python object.

    To save time during debugging sessions, you can manually get
    `mpii_dataset_mat` using `scipy.io.loadmat` once, and then iteratively call
    this function as you make changes, without reloading the .mat file.

    Args:
        mpii_dataset_mat: A dictionary of MATLAB structures loaded using
            `scipy.io.loadmat`. The arguments `struct_as_record = False` and
            `squeeze_me = True` must be set in the `loadmat` call.
        mpii_images_dir: The path of the directory where all the MPII images
            are stored.

    Returns: An `MpiiDataset` Python object correspodning to
        `mpii_dataset_mat`.
    """
    mpii_annotations = mpii_dataset_mat.annolist
    train_or_test = mpii_dataset_mat.img_train

    img_filenames = []
    people_in_imgs = []
    for img_index in range(len(mpii_annotations)):
        if train_or_test[img_index] == 1:
            img_abs_filepath, people = _parse_annotation(mpii_annotations[img_index],
                                                         mpii_images_dir)

            # NOTE(brendan): There are annotations in the MPII dataset for
            # which the file corresponding to image.name does not exist.
            # Therefore we have to check that the image is present before
            # adding it to our structure.
            if not os.path.exists(img_abs_filepath):
                continue

            img_filenames.append(img_abs_filepath)
            people_in_imgs.append(people)

    return MpiiDataset(img_filenames, people_in_imgs)

def mpii_read(mpii_dataset_filepath):
    """
    Note that the images are assumed to reside in a folder one up from the .mat
    file that is being parsed (i.e. ../images).

    Args:
        mpii_dataset_filepath: The filepath to the .mat file provided from the
            MPII Human Pose website.

    Returns: Parsed `MpiiDataset` object from `parse_mpii_data_from_mat`.
    """
    mpii_dataset_mat = scipy.io.loadmat(mpii_dataset_filepath,
                                        struct_as_record=False,
                                        squeeze_me=True)['RELEASE']

    mpii_dataset_dir = os.path.dirname(mpii_dataset_filepath)
    mpii_images_dir = os.path.join(mpii_dataset_dir, '../images')

    return parse_mpii_data_from_mat(mpii_dataset_mat, mpii_images_dir)

if __name__ == "__main__":
    assert len(sys.argv) == 1

    mpii_dataset = mpii_read(sys.argv[0])
