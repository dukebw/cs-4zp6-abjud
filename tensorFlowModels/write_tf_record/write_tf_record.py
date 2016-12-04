from PIL import Image, ImageDraw
from mpii_read import mpii_read
import tensorflow as tf

def clamp01(value):
    return max(0, min(value, 1))

def show_image_with_joints(next_image, people_in_img):
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

    for img_index in range(4):
        next_image = session.run(img_tensor)
        show_image_with_joints(next_image,
                               mpii_dataset.people_in_imgs[img_index])

    coord.request_stop()
    coord.join(threads)

def main(argv):
    assert len(argv) == 1

    mpii_dataset = mpii_read(argv[0])

    write_tf_record(mpii_dataset)

if __name__ == "__main__":
    tf.app.run()
