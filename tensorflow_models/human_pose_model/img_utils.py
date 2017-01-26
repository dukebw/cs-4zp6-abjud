from PIL import Image, ImageDraw
import numpy as np

''' Example:
    head = df.head_rect[i]
    joints = df.joint_pos[i]
    image = df.filename[i]
'''
def draw(image, head, joints, is_visible):
    '''Descr: Draws filled, red circles on top of visible joints, white nonfilled circles on top of non-visible joints and a red circle around the head.
    Input: image, head (flattened vector), joints (numpy array) and is_visible (dictionary)
    Output: Image with joints and head marked.
    '''
    draw = ImageDraw.Draw(image)
    im = Image.fromarray(np.asarray(image))
    draw.rectangle(head,outline=(255,0,0))
    for i in xrange(len(joints)):
        if is_visible[str(i)]:
            draw.ellipse((joints[i][0]-7, joints[i][1]-7,joints[i][0]+7,joints[i][1]+7), fill=(255,0,0))
        else:
            draw.ellipse((joints[i][0]-7, joints[i][1]-7,joints[i][0]+7,joints[i][1]+7),fill=None, outline =(255,0,0))

    image.show()


def img2array(img):
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype='float32')
    return x

