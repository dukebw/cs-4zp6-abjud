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
`annolist[index].annorect.annopoints.point` field. Coordinates of the objpos
and scale can be found from `annolist[index].annorect.objpos` and
annolist[index].annorect.scale, respectively.

Out of the total dataset of 24987 images, 17408 of those are usable training
images, meaning that they have at least one person with joints labelled,
`objpos.{x, y}`, and `scale`. This is the number of images returned by the
`mpii_read` function for training images. 6619 examples are returned for test.

Test images do not seem to have head rectangles or joint annotations, and
rather only contain the `objpos` and `scale` values as methods of estimating
where the person is in the picture.

15247 of the training images contain annotations for single people, indicated
by the `single_person` list.

Each joint has an `is_visible` attribute, indicating whether it is visible or
occluded.
"""
import sys
import os
import scipy.io
import numpy as np
from dataset.shapes import Rectangle
from dataset.mpii_datatypes import Person, MpiiDataset

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


def _parse_annotation(img_annotation,
                      single_person_list,
                      mpii_images_dir,
                      is_train):
    """Parses a single image annotation from the MPII dataset.

    Looks at the annotations for a single image, and returns the people in the
    image along with the full filepath of the image.

    Args:
        img_annotation: The annotations coming from annolist(index) from the
            MPII dataset.
        single_person_list: List of MATLAB indices (starts from 1) of singular
            people in this image.
        mpii_images_dir: Path to the directory where the MPII images are.
        is_train: Training or test annotation? Training annotations require at
            least one joint to be annotated in order to be useful.

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
    for annorect_index in single_person_list:
        img_annorect = img_annotation.annorect[annorect_index - 1]

        try:
            objpos = img_annorect.objpos
            if (not hasattr(objpos, 'x')) or (not hasattr(objpos, 'y')):
                continue

            scale = img_annorect.scale
            assert scale > 0
        except AttributeError:
            continue

        try:
            head_rect = Rectangle((img_annorect.x1, img_annorect.y1,
                                   img_annorect.x2, img_annorect.y2))
            person = Person(img_annorect.annopoints.point,
                            objpos,
                            scale,
                            head_rect)
            people.append(person)
        except AttributeError:
            if is_train:
                continue

            people.append(Person([], objpos, scale))

    return img_abs_filepath, people


def _shuffle_list(list_l, shuffled_indices):
    """Shuffles list_l by re-ordering the list based on the indices in
    shuffled_indices.

    Args:
        list_l: List to be shuffled.
        shuffled_indices: Set of indices containing [0:len(list_l)), which is
            assumed to have already been shuffled.

    Returns:
        List with all the same elements as `list_l`, in the shuffled order given
        by `shuffled_indices`.
    """
    return [list_l[index] for index in shuffled_indices]


def _shuffle_dataset(img_filenames, people_in_imgs):
    """Shuffles the list of filenames and labels in the MPII dataset.

    Args:
        img_filenames: List of filenames in the MPII dataset to be shuffled.
        people_in_imgs: List of `Person`s in the MPII dataset, to be shuffled.
    """
    img_indices = list(range(len(img_filenames)))
    np.random.shuffle(img_indices)

    img_filenames = _shuffle_list(img_filenames, img_indices)
    people_in_imgs = _shuffle_list(people_in_imgs, img_indices)

    return img_filenames, people_in_imgs


def parse_mpii_data_from_mat(mpii_dataset_mat, mpii_images_dir, is_train):
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
        is_train: Parse training data (True), or test data (False)?

    Returns: An `MpiiDataset` Python object correspodning to
        `mpii_dataset_mat`.
    """
    mpii_annotations = mpii_dataset_mat.annolist
    train_or_test = mpii_dataset_mat.img_train

    img_filenames = []
    people_in_imgs = []
    filenames_on_disk = set(os.listdir(mpii_images_dir))
    for img_index in range(len(mpii_annotations)):
        if train_or_test[img_index] == int(is_train):
            single_person_list = _make_iterable(mpii_dataset_mat.single_person[img_index])
            img_abs_filepath, people = _parse_annotation(mpii_annotations[img_index],
                                                         single_person_list,
                                                         mpii_images_dir,
                                                         is_train)
            if len(people) == 0:
                continue

            # NOTE(brendan): There are annotations in the MPII dataset for
            # which the file corresponding to image.name does not exist.
            # Therefore we have to check that the image is present before
            # adding it to our structure.
            if not os.path.basename(img_abs_filepath) in filenames_on_disk:
                continue

            img_filenames.append(img_abs_filepath)
            people_in_imgs.append(people)

    img_filenames, people_in_imgs = _shuffle_dataset(img_filenames,
                                                     people_in_imgs)

    return MpiiDataset(img_filenames, people_in_imgs)


def mpii_read(mpii_dataset_filepath, is_train):
    """
    Note that the images are assumed to reside in a folder one up from the .mat
    file that is being parsed (i.e. ../images).

    Args:
        mpii_dataset_filepath: The filepath to the .mat file provided from the
            MPII Human Pose website.
        is_train: Read training data (True), or test data (False)?

    Returns: Parsed `MpiiDataset` object from `parse_mpii_data_from_mat`.
    """
    mpii_dataset_mat = scipy.io.loadmat(mpii_dataset_filepath,
                                        struct_as_record=False,
                                        squeeze_me=True)['RELEASE']

    mpii_dataset_dir = os.path.dirname(mpii_dataset_filepath)
    mpii_images_dir = os.path.join(mpii_dataset_dir, '../images')

    return parse_mpii_data_from_mat(mpii_dataset_mat, mpii_images_dir, is_train)


if __name__ == "__main__":
    assert len(sys.argv) == 1

    mpii_dataset = mpii_read(sys.argv[0])
