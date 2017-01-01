import os
import time
import copy
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
import mpii_read

FLAGS = tf.app.flags.FLAGS

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 18000

RMSPROP_DECAY = 0.9
RMSPROP_MOMENTUM = 0.9
RMSPROP_EPSILON = 1.0

NUM_JOINTS = 16
NUM_JOINT_COORDS = 2*NUM_JOINTS

tf.app.flags.DEFINE_integer('image_dim', 299,
                            """Dimension of the square input image.""")

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of threads to use to preprocess
                            images.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of threads to use to read example
                            protobufs from TFRecords.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """Number of GPUs in system.""")

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Size of each mini-batch (number of examples
                            processed at once).""")
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Factor by which to increase the minimum examples
                            in RandomShuffleQueue.""")

tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Rate at which learning rate is decayed.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Number of epochs before decay factor is applied
                          once.""")

def main(argv=None):
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            # TODO(brendan): Support multiple GPUs?
            assert FLAGS.num_gpus == 1

            with tf.name_scope('batch_processing'):
                data_filenames = tf.gfile.Glob('./train*tfrecord')
                assert data_filenames, ('No data files found.')
                assert len(data_filenames) >= FLAGS.num_readers

                filename_queue = tf.train.string_input_producer(
                    string_tensor=data_filenames,
                    capacity=16)

                # TODO(brendan): Alter `write_tf_record` code to spit out
                # shards with about 1024 examples each.
                examples_per_shard = 1024
                min_queue_examples = FLAGS.input_queue_memory_factor*examples_per_shard

                examples_queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3*FLAGS.batch_size,
                                                       min_after_dequeue=min_queue_examples,
                                                       dtypes=[tf.string])

                # TODO(brendan): FLAGS.num_readers == 1 case
                assert FLAGS.num_readers > 1

                enqueue_ops = []
                for _ in range(FLAGS.num_readers):
                    reader = tf.TFRecordReader()
                    _, per_thread_example = reader.read(queue=filename_queue)
                    enqueue_ops.append(examples_queue.enqueue(vals=[per_thread_example]))

                tf.train.queue_runner.add_queue_runner(
                    qr=tf.train.queue_runner.QueueRunner(queue=examples_queue,
                                                         enqueue_ops=enqueue_ops))
                example_serialized = examples_queue.dequeue()

                images_and_joints = []
                for thread_id in range(FLAGS.num_preprocess_threads):
                    feature_map = {
                        'image_jpeg': tf.FixedLenFeature(shape=[], dtype=tf.string),
                        'joint_indices': tf.VarLenFeature(dtype=tf.int64),
                        'joints': tf.VarLenFeature(dtype=tf.float32)
                    }
                    features = tf.parse_single_example(
                        serialized=example_serialized, features=feature_map)

                    img_jpeg = features['image_jpeg']
                    with tf.name_scope(name='decode_jpeg', values=[img_jpeg]):
                        img_tensor = tf.image.decode_jpeg(contents=img_jpeg,
                                                          channels=3)
                        decoded_img = tf.image.convert_image_dtype(
                            image=img_tensor, dtype=tf.float32)

                    # TODO(brendan): Image distortion goes here.

                    decoded_img = tf.sub(x=decoded_img, y=0.5)
                    decoded_img = tf.mul(x=decoded_img, y=2.0)

                    decoded_img = tf.reshape(
                        tensor=decoded_img,
                        shape=[FLAGS.image_dim, FLAGS.image_dim, 3])

                    images_and_joints.append([decoded_img,
                                              features['joint_indices'],
                                              features['joints']])

                images, joint_indices, joints = tf.train.batch_join(
                    tensors_list=images_and_joints,
                    batch_size=FLAGS.batch_size,
                    capacity=2*FLAGS.num_preprocess_threads*FLAGS.batch_size)

                tf.summary.image(name='images', tensor=images)

            input_summaries = copy.copy(tf.get_collection(key=tf.GraphKeys.SUMMARIES))

            global_step = tf.get_variable(
                name='global_step',
                shape=[],
                initializer=tf.constant_initializer(0),
                trainable=False)

            num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                     FLAGS.batch_size)
            decay_steps = int(num_batches_per_epoch*FLAGS.num_epochs_per_decay)
            learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.initial_learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=decay_steps,
                                                       decay_rate=FLAGS.learning_rate_decay_factor)

            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                  decay=RMSPROP_DECAY,
                                                  momentum=RMSPROP_MOMENTUM,
                                                  epsilon=RMSPROP_EPSILON)

            with tf.device(device_name_or_function='/gpu:0'):
                with tf.name_scope(name='tower0') as scope:
                    with slim.arg_scope([slim.model_variable], device='/cpu:0'):
                        with slim.arg_scope(inception.inception_v3_arg_scope()):
                            logits, endpoints = inception.inception_v3(inputs=images,
                                                                       num_classes=32,
                                                                       scope=scope)

                            with tf.name_scope('summaries'):
                                for activation in endpoints.values():
                                    tensor_name = activation.op.name
                                    tf.summary.histogram(
                                        name=tensor_name + '/activations',
                                        values=activation)
                                    tf.summary.scalar(
                                        name=tensor_name + '/sparsity',
                                        tensor=tf.nn.zero_fraction(value=activation))

                            auxiliary_logits = endpoints['AuxLogits']

                            sparse_joints = tf.sparse_merge(sp_ids=joint_indices,
                                                            sp_values=joints,
                                                            vocab_size=NUM_JOINT_COORDS)
                            dense_joints = tf.sparse_tensor_to_dense(sp_input=sparse_joints,
                                                                     default_value=0)

                            dense_shape = [FLAGS.batch_size, NUM_JOINT_COORDS]
                            dense_joints = tf.reshape(tensor=dense_joints, shape=dense_shape)

                            weights = tf.sparse_to_dense(sparse_indices=sparse_joints.indices,
                                                         output_shape=dense_shape,
                                                         sparse_values=1,
                                                         default_value=0)

                            slim.losses.mean_squared_error(predictions=logits,
                                                           labels=dense_joints,
                                                           weights=weights)
                            slim.losses.mean_squared_error(predictions=auxiliary_logits,
                                                           labels=dense_joints,
                                                           weights=weights,
                                                           scope='aux_logits')

                            total_loss = slim.losses.get_total_loss()

                            # TODO(brendan): Calculate loss averages for tensorboard

                    summaries = tf.get_collection(key=tf.GraphKeys.SUMMARIES, scope=scope)

                    batchnorm_updates = tf.get_collection(
                        key=tf.GraphKeys.UPDATE_OPS, scope=scope)

                    gradients = optimizer.compute_gradients(loss=total_loss)

            summaries.extend(input_summaries)

            summaries.append(
                tf.summary.scalar(name='learning_rate', tensor=learning_rate))

            for grad, var in gradients:
                if grad is not None:
                    summaries.append(
                        tf.summary.histogram(name=var.op.name + '/gradients',
                                             values=grad))

            apply_gradient_op = optimizer.apply_gradients(
                grads_and_vars=gradients, global_step=global_step)

            for var in tf.trainable_variables():
                summaries.append(
                    tf.summary.histogram(name=var.op.name, values=var))

            # TODO(brendan): track moving averages of trainable variables

            batchnorm_updates_op = tf.group(*batchnorm_updates)
            train_op = tf.group(batchnorm_updates_op, apply_gradient_op)

            saver = tf.train.Saver(var_list=tf.global_variables())

            summary_op = tf.summary.merge(inputs=summaries)

            init = tf.global_variables_initializer()

            session = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True))
            session.run(init)

            # TODO(brendan): check for pre-trained checkpoint

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord)

            log_dir = '../../log'
            summary_writer = tf.summary.FileWriter(logdir=log_dir,
                                                   graph=session.graph)

            for step in range(1000000):
                start = time.perf_counter()
                _, loss_value = session.run(fetches=[train_op, total_loss])
                end = time.perf_counter()
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    print('step {}, loss = {}'.format(step, loss_value))

                if step % 100 == 0:
                    summary_string = session.run(fetches=summary_op)
                    summary_writer.add_summary(summary=summary_string)

                if step % 5000 == 0:
                    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess=session,
                               save_path=checkpoint_path,
                               global_step=step)

            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
