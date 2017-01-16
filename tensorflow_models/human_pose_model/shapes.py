"""This module contains definitions of useful geometric shapes.
"""

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


