import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
from input_pipeline import setup_eval_input_pipeline

# @debug
from PIL import Image, ImageDraw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '.',
                           """Path to take input TFRecord files from.""")
tf.app.flags.DEFINE_integer('image_dim', 299,
                            """Dimension of the square input image.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of threads to use to preprocess
                            images.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Size of each mini-batch (number of examples
                            processed at once).""")

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            eval_batch = setup_eval_input_pipeline(FLAGS.data_dir,
                                                   FLAGS.batch_size,
                                                   FLAGS.num_preprocess_threads,
                                                   FLAGS.image_dim)

            num_joint_coords = 32
            with tf.name_scope(name='tower0') as scope:
                with slim.arg_scope([slim.model_variable], device='/cpu:0'):
                    with slim.arg_scope(inception.inception_v3_arg_scope()):
                        logits, _ = inception.inception_v3(inputs=eval_batch.images,
                                                           num_classes=num_joint_coords,
                                                           scope=scope)

            restorer = tf.train.Saver()

            session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            restorer.restore(sess=session, save_path='./log/jan2-2017-run2/model.ckpt-2109')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            converted_imgs = tf.div(x=eval_batch.images, y=2.0)
            converted_imgs = tf.add(x=converted_imgs, y=0.5)

            converted_imgs = tf.image.convert_image_dtype(
                image=converted_imgs, dtype=tf.uint8)

            [next_images, predictions] = session.run(
                    fetches=[converted_imgs, logits])

            for img_index in range(len(next_images)):
                pil_image = Image.fromarray(next_images[img_index])
                draw = ImageDraw.Draw(pil_image)

                joints = 299*((predictions[img_index]/2) + 0.5)
                for joint_index in range(16):
                    x = joints[joint_index]
                    y = joints[joint_index + 1]
                    box = (x - 5, y - 5, x + 5, y + 5)
                    colour = (0xFF, 0, 0)

                    draw.ellipse(box, fill=colour)

                pil_image.show()

            coord.request_stop()
            coord.join(threads)


def main(argv=None):
    evaluate()


if __name__ == "__main__":
    tf.app.run()
