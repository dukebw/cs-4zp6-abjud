"""
A module for reading the [MPII Human Pose
Dataset](http://human-pose.mpi-inf.mpg.de/).
"""
import sys
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
    mpii_dataset_mat = scipy.io.loadmat(mpii_dataset_filepath)
    mpii_annotations = mpii_dataset_mat['RELEASE']['annolist'][0, 0]
    images = mpii_annotations['image']

    for img_index in range(mpii_annotations.shape[1]):
        img_filename = images[0, img_index][0, 0]['name'][0]

if __name__ == "__main__":
    assert len(sys.argv) == 1

    mpii_read(sys.argv[0])
