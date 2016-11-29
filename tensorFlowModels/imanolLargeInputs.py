# TensorFlow Input Pipelines for Large Data Sets
# ischlag.github.io
# TensorFlow 0.11, 07.11.2016

import tensorflow as tf
import numpy as np
import threading
import math


# Generating some simple data
r = np.arange(0.0,100003.0)
#print(r.shape)

# so there's set of r,r,r,r -> r is defined above
# and so you get 
raw_data = np.dstack((r,r,r,r))[0]
data_size_label = 25
data_size_input = 4
num_in_batch = 20

gauss_deviation = 10
norm_factor = 2*math.pi*gauss_deviation*gauss_deviation
norm_factor = 1/norm_factor
#
#print(raw_data.shape)
raw_target = np.array([[1,0,0]] * 100003)

# are used to feed data into our queue
queue_input_data = tf.placeholder(tf.float32, shape=[num_in_batch, data_size_input])
queue_input_target = tf.placeholder(tf.float32, shape=[num_in_batch, data_size_label])

queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.float32], shapes=[[data_size_input], [data_size_label]])

enqueue_op = queue.enqueue_many([queue_input_data, queue_input_target])
dequeue_op = queue.dequeue()

# tensorflow recommendation:
# capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=15, capacity=40)
# use this to shuffle batches:
# data_batch, target_batch = tf.train.shuffle_batch(dequeue_op, batch_size=15, capacity=40, min_after_dequeue=5)

def encode_gauss_map(x_avg, y_avg): # here I should create a 64x64 sized array
    labels = []
    for x in range(1, 6):
        for y in range(1, 6):
            value = 2*gauss_deviation**2
            value = ((x - x_avg)**2 + (y - y_avg)**2)/value
            value = norm_factor * math.e **(-value)
            labels.append(value)
    return labels


def enqueue(sess):
  """ Iterates over our data puts small junks into our queue."""
  under = 0
  max = len(raw_data)
  print(max)
  print(under+20)
  while True:
    print("starting to write into queue")
    upper = under + 20
    print("try to enqueue ", under, " to ", upper)
    if upper <= max:
      #print(curr_target.shape)
      # create a (20,25) numpy array and the create gauss map only
      # creates a (25) =(5, 5) array
      curr_target = np.empty([20,25])
      for x in range(0,20):
          curr_target[x,:] = np.array(encode_gauss_map(1,1))
      curr_data = raw_data[under:upper]
      print(curr_target.shape)
      under = upper
    '''
    else:
      rest = upper - max
      curr_data = np.concatenate((raw_data[under:max], raw_data[0:rest]))
      curr_target = np.concatenate((raw_target[under:max], raw_target[0:rest]))
      under = rest
    '''
    sess.run(enqueue_op, feed_dict={queue_input_data: curr_data,
                                    queue_input_target: curr_target})
    print("added to the queue")
  print("finished enqueueing")

# start the threads for our FIFOQueue and batch
sess = tf.Session()
enqueue_thread = threading.Thread(target=enqueue, args=[sess])
enqueue_thread.isDaemon()
enqueue_thread.start()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# Fetch the data from the pipeline and put it where it belongs (into your model)
for i in range(5):
  run_options = tf.RunOptions(timeout_in_ms=4000)
  curr_data_batch, curr_target_batch = sess.run([data_batch, target_batch], options=run_options)
  print(curr_data_batch)

# shutdown everything to avoid zombies
sess.run(queue.close(cancel_pending_enqueues=True))
coord.request_stop()
coord.join(threads)
sess.close()


