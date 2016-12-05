from PIL import Image, ImageDraw
from mpii_read import mpii_read
import tensorflow as tf

def clamp01(value):
    """Clamps value to the range [0.0, 1.0].

    Args:
        value: A number to be clamped.

    Return:
        value if value is in [0.0, 1.0], otherwise whichever of 0.0 or 1.0 is
        closer to value.
    """
    return max(0, min(value, 1))

def show_image_with_joints(next_image, people_in_img):
    """Debug function to display an image with the head rectangle and joints.

    Args:
        next_image: The image to display.
        people_in_img: the list of `Person`s corresponding to next_image from
            the MpiiDataset class.
    """
    image = Image.fromarray(next_image)
    draw = ImageDraw.Draw(image)

    for person in people_in_img:
        draw.rectangle(person.head_rect, outline = (0, 0xFF, 0))

        for joint_index in range(len(person.joints)):
            joint = person.joints[joint_index]

            if joint:
                red = int(0xFF*(joint_index % 5)/5)
                green = int(0xFF*(joint_index % 10)/10)
                blue = int(0xFF*joint_index/16)
                colour = (red, green, blue)
                box = (joint[0] - 5, joint[1] - 5,
                       joint[0] + 5, joint[1] + 5)
                draw.ellipse(box, fill = colour)

    image.show()

def write_tf_record(mpii_dataset):
    session = tf.InteractiveSession()

    filename_queue = tf.train.string_input_producer(mpii_dataset.img_filenames,
                                                    shuffle = False)
    _, img_jpeg = tf.WholeFileReader().read(filename_queue)
    img_tensor = tf.image.decode_jpeg(img_jpeg)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    for img_index in range(len(mpii_dataset.img_filenames)):
        next_image = session.run(img_tensor)

    coord.request_stop()
    coord.join(threads)

def main(argv):
    assert len(argv) == 1

    mpii_dataset = mpii_read(argv[0])

    write_tf_record(mpii_dataset)

if __name__ == "__main__":
    tf.app.run()
