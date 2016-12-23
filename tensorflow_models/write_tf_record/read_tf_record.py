from PIL import Image, ImageDraw
import tensorflow as tf
import mpii_read

INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
NUM_EPOCHS_PER_DECAY = 350
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
BATCH_SIZE = 128

def main(argv=None):
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            ### training stuff start

            global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                trainable=False)

            num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                     BATCH_SIZE)
            decay_steps = int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)
            learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                       global_step,
                                                       decay_steps,
                                                       LEARNING_RATE_DECAY_FACTOR)

            optimizer = tf.train.RMSPropOptimizer(learning_rate)

            ### training stuff end

            # TODO(brendan): make filename queue operate on the different
            # TFRecord chunks.
            filename_queue = tf.train.string_input_producer(
                ['train0.tfrecord'],
                capacity=16)

            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)

            feature_map = {
                'image_jpeg': tf.FixedLenFeature([], tf.string),
                'joint_bitmaps': tf.VarLenFeature(tf.int64),
                'joints': tf.VarLenFeature(tf.float32)
            }
            features = tf.parse_single_example(example_serialized, feature_map)
            img_tensor = tf.image.decode_jpeg(features['image_jpeg'],
                                              channels=3)

            init = tf.initialize_all_variables()

            session = tf.Session()
            session.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord)

            image_dim = 220
            image_center = 110
            for _ in range(4):
                [image, joints, joint_bitmaps] = session.run(
                    [img_tensor, features['joints'], features['joint_bitmaps']])

                pil_image = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_image)

                sparse_joint_index = 0
                for joint_bitmap in joint_bitmaps.values:
                    joint_index = 0
                    while joint_bitmap > 0:
                        if (joint_bitmap & 0x1) == 0x1:
                            red = int(0xFF*(joint_index % 5)/5)
                            green = int(0xFF*(joint_index % 10)/10)
                            blue = int(0xFF*joint_index/16)
                            colour = (red, green, blue)

                            x = joints.values[sparse_joint_index]
                            x_scaled = int(x*image_dim + image_center)
                            y = joints.values[sparse_joint_index + 1]
                            y_scaled = int(y*image_dim + image_center)
                            box = (x_scaled - 2, y_scaled - 2,
                                   x_scaled + 2, y_scaled + 2)
                            draw.ellipse(box, colour)

                            sparse_joint_index += 2

                        joint_bitmap >>= 1
                        joint_index += 1

                pil_image.show()

            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
