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
import cv2
import scipy.io

class Joints(object):
    pass

class MpiiDataset(object):
    """
    Representation of the entire MPII Dataset.

    The annotation description can be found
    [here](http://human-pose.mpi-inf.mpg.de/#download).

    Currently only the images and person-centric body joint annotations are
    taken from the dataset.
    """
    def __init__(self, images, joints):
        self._images = images
        self._joints = joints

    @property
    def images(self):
        return self._images

    @property
    def joints(self):
        return self._joints

def mpii_read(mpii_dataset_filepath):
    """
    Note that the images are assumed to reside in a folder one up from the .mat
    file that is being parsed (i.e. ../images).
    """
    mpii_dataset_mat = scipy.io.loadmat(mpii_dataset_filepath)['RELEASE']
    mpii_annotations = mpii_dataset_mat['annolist'][0, 0]
    images = mpii_annotations['image']
    train_or_test = mpii_dataset_mat['img_train'][0, 0]
    annorect = mpii_annotations['annorect']

    for img_index in range(mpii_annotations.shape[1]):
        if train_or_test[0, img_index] == 1:
            img_filename = images[0, img_index][0, 0]['name'][0]
            mpii_dataset_dir = os.path.dirname(mpii_dataset_filepath)
            img_abs_filepath = os.path.join(mpii_dataset_dir,
                                            '../images',
                                            img_filename)

            image = cv2.imread(img_abs_filepath)

            img_annorect = annorect[0, img_index]
            for person_index in range(img_annorect['x1'].shape[1]):
                x1 = img_annorect['x1'][0, person_index][0, 0]
                y1 = img_annorect['y1'][0, person_index][0, 0]
                x2 = img_annorect['x2'][0, person_index][0, 0]
                y2 = img_annorect['y2'][0, person_index][0, 0]

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0xFF, 0), 3)

            cv2.imshow('image', image)
            cv2.waitKey(0)

if __name__ == "__main__":
    assert len(sys.argv) == 1

    mpii_read(sys.argv[0])
