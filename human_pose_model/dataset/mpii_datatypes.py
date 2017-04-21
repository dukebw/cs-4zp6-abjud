"""This module contains the datatype definitions particular to the MPII
dataset.
"""

JOINT_NAMES = ['0 - r ankle',
               '1 - r knee',
               '2 - r hip',
               '3 - l hip',
               '4 - l knee',
               '5 - l ankle',
               '6 - pelvis',
               '7 - thorax',
               '8 - upper neck',
               '9 - head top',
               '10 - r wrist',
               '11 - r elbow',
               '12 - r shoulder',
               '13 - l shoulder',
               '14 - l elbow',
               '15 - l wrist']

class Joint(object):
    """Class to represent a joint, including x and y position and `is_visible`
    indicating whether the joint is visible or occluded.
    """
    def __init__(self, x, y, is_visible):
        self._x = x
        self._y = y
        self._is_visible = is_visible

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def is_visible(self):
        return self._is_visible


class Person(object):
    """A class representing each person in a given image, including their
    joints, objpos and scale.

    The joints should be a list of (x, y) tuples where x and y are both in the
    range [img_x_max, img_y_max], and the joint ids are as follows,

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
        objpos: The approximate position of the center of the person in the
            image.
        scale: Scale of the person with respect to 200px.
    """
    NUM_JOINTS = 16

    def __init__(self, joints, objpos, scale, head_rect):
        self._joints = Person.NUM_JOINTS*[None]

        if not hasattr(joints, '__iter__'):
            joints = [joints]

        for joint in joints:
            # NOTE(brendan): Only certain joints have the `is_visible`
            # annotation, and some images have no `is_visible` annotations at
            # all. Since the majority of joints are visible, we convert
            # unannotated joints to visible.
            # An experiment would be to try the opposite and compare results.
            if (type(joint.is_visible) is not int):
                is_visible = 1
            else:
                is_visible = joint.is_visible

            self._joints[joint.id] = Joint(joint.x, joint.y, is_visible)

        self._objpos = objpos
        self._scale = scale
        self._head_rect = head_rect

    @property
    def joints(self):
        return self._joints

    @property
    def objpos(self):
        return self._objpos

    @property
    def scale(self):
        return self._scale

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
        assert len(img_filenames) == len(people_in_imgs)

        self._img_filenames = img_filenames
        self._people_in_imgs = people_in_imgs

    @property
    def img_filenames(self):
        return self._img_filenames

    @property
    def people_in_imgs(self):
        return self._people_in_imgs
