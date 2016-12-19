import tensorflow as tf

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

            init = tf.initialize_all_variables()

            session = tf.Session()
            session.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord)

            # TODO(brendan): Make sense of the `SparseTensorValue` returned
            # from evaluating `features`, parse out the joint locations
            # properly, and draw the joints.
            feature_map = {
                'image_jpeg': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'joints_bitmaps': tf.VarLenFeature(tf.int64),
                'joints': tf.VarLenFeature(tf.float32)
            }
            features = tf.parse_single_example(example_serialized, feature_map)
            img_tensor = tf.image.decode_jpeg(features['image_jpeg'],
                                              channels=3)

            image = session.run(img_tensor)

            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    tf.app.run()
