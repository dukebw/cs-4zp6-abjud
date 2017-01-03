import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            eval_batch = setup_eval_input_pipeline(data_dir,
                                                   batch_size,
                                                   num_preprocess_threads,
                                                   image_dim)

            num_joint_coords = 32
            logits, _ = inception.inception_v3(inputs=eval_batch.images,
                                               num_classes=num_joint_coords)

            # TODO(brendan): Load previous model checkpoint, get predictions
            # and draw images.


def main(argv=None):
    evaluate()


if __name__ == "__main__":
    tf.app.run()
