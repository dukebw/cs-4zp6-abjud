"""This module contains standalone utility functions that can be used across
other modules.
"""
import os
import threading
import numpy as np
import tensorflow as tf

def get_n_ranges(start, end, num_ranges):
    """Takes a start index, end index, and number of ranges, e.g. start=0,
    end=2 and num_ranges=2, and returns a list of lists indicating
    evenly-spaced ranges, i.e.  [[0, 1], [1, 2]].
    """
    spacing = np.linspace(start, end, num_ranges + 1).astype(np.int)

    ranges = []
    for spacing_index in range(len(spacing) - 1):
        ranges.append([spacing[spacing_index], spacing[spacing_index + 1]])

    return ranges


def _thread_count_examples(num_examples_results,
                           train_data_filenames,
                           ranges,
                           thread_index):
    """Per-thread function to count the number of examples in its given range
    (`ranges[thread_index]`) of the `train_data_filenames` list.

    The resultant count is returned in `num_examples_results[thread_index]`,
    such that `num_examples_results` can be summed after all the threads are
    joined, in order to produce the total number of examples.
    """
    options = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
    for file_index in range(ranges[thread_index][0], ranges[thread_index][1]):
        data_file = train_data_filenames[file_index]
        for _ in tf.python_io.tf_record_iterator(path=data_file, options=options):
            num_examples_results[thread_index] += 1


def count_training_examples(data_dir, num_threads, tfrecord_prefix):
    """Counts training examples in the TFRecords given by
    `train_data_filenames`, by creating `num_threads` threads, which will all
    count the examples in roughly 1/train_data_filenames of the files.
    """
    data_filenames = tf.gfile.Glob(
        os.path.join(data_dir, tfrecord_prefix + '*tfrecord'))
    assert data_filenames, ('No data files found.')

    coord = tf.train.Coordinator()

    ranges = get_n_ranges(0, len(data_filenames), num_threads)

    num_examples_results = num_threads*[0]
    threads = []
    for thread_index in range(num_threads):
        args = (num_examples_results, data_filenames, ranges, thread_index)
        t = threading.Thread(target=_thread_count_examples, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)

    return sum(num_examples_results), data_filenames
