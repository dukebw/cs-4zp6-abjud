import tensorflow as tf

INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
NUM_EPOCHS_PER_DECAY = 350
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
BATCH_SIZE = 128

def main(argv=None):
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                trainable=False)

            num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                     BATCH_SIZE)
            decay_steps = int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)
            learning_rate = tf.train.exponential_decay(learning_rate,
                                                       global_step,
                                                       decay_steps,
                                                       LEARNING_RATE_DECAY_FACTOR)

            optimizer = tf.train.RMSPropOptimizer(learning_rate)

            # TODO(brendan): make filename queue with string_input_producer

if __name__ == "__main__":
    tf.app.run()
