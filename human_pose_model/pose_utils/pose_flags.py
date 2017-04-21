import math
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('network_name', 'resnet_bulat',
                           """Name of desired network to use for part
                           detection. Valid options: vgg, inception_v3.""")

tf.app.flags.DEFINE_string('loss_name', 'mean_squared_error_loss',
                           """Name of desired loss function to use.""")

tf.app.flags.DEFINE_string('train_data_dir',
                           '/tmp/MPII_Data/train_80_shards_w_binmap',
                           """Path to take input training TFRecord files
                           from.""")

tf.app.flags.DEFINE_string('validation_data_dir',
                           '/tmp/MPII_Data/valid_80_shards_w_binmap',
                           """Path to take input validation TFRecord files
                           from.""")

tf.app.flags.DEFINE_string('log_dir', 'logs/resnet/exp0',
                           """Path to take summaries and checkpoints from, and
                           write them to.""")

tf.app.flags.DEFINE_string('log_filename', 'train_log',
                           """Name of file to log training steps and loss
                           to.""")

tf.app.flags.DEFINE_string('checkpoint_path',
                           None,
                           """Path to take checkpoint file (e.g.
                           inception_v3.ckpt) from.""")

tf.app.flags.DEFINE_string('second_checkpoint_path',
                           None,
                           """Regression subnetwork path to take checkpoint
                           file (e.g.  inception_v3.ckpt) from.""")

tf.app.flags.DEFINE_string('checkpoint_exclude_scopes',
                           None,
                           """Comma-separated list of scopes to exclude when
                           restoring from a checkpoint.""")

tf.app.flags.DEFINE_string('second_checkpoint_exclude_scopes',
                           None,
                           """Comma-separated list of scopes to exclude when
                           restoring the regression subnetwork from a
                           checkpoint.""")

tf.app.flags.DEFINE_string('trainable_scopes', None,
                           """Comma-separated list of scopes to train.""")

tf.app.flags.DEFINE_integer('image_dim', 256,
                            """Dimension of the square input image.""")

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of threads to use to preprocess
                            images.""")

tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of threads to use to read example
                            protobufs from TFRecords.""")

tf.app.flags.DEFINE_integer('num_gpus', 4,
                            """Number of GPUs in system.""")

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Size of each mini-batch (number of examples
                            processed at once).""")

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 4,
                            """Factor by which to increase the minimum examples
                            in RandomShuffleQueue.""")

tf.app.flags.DEFINE_integer('max_epochs', 90,
                            """Maximum number of epochs in training run.""")

tf.app.flags.DEFINE_integer('heatmap_stddev_pixels', 5,
                            """Standard deviation of Gaussian joint heatmap, in pixels.""")

tf.app.flags.DEFINE_integer('eval_interval_secs', 360,
                            """Interval in seconds for which we will wait
                            between checking for new checkpoints and evaluating
                            them.""")

tf.app.flags.DEFINE_float('max_rotation_angle', math.pi/6,
                          """Maximum amount to rotate images, in radians.""")

tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
                          """Initial learning rate.""")

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Rate at which learning rate is decayed.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Number of epochs before decay factor is applied
                          once.""")

tf.app.flags.DEFINE_boolean('is_detector_training', False,
                            """Set to true if the detector subnetwork should be
                            trained.""")

tf.app.flags.DEFINE_boolean('restore_global_step', True,
                            """Set to True if restoring a training run that is
                            part-way complete.""")

tf.app.flags.DEFINE_boolean('is_regression_subnetwork_pretrained', False,
                            """Set to True if restoring ILSVRC pre-trained
                            weights to the regression subnetwork.""")

tf.app.flags.DEFINE_string('optimizer', None,
                           """optimizer to use. None for RMSProp 'adam' for adam optimizer""")

tf.app.flags.DEFINE_string('checkpoint_name', None,
                           """Name of checkpoint""")
