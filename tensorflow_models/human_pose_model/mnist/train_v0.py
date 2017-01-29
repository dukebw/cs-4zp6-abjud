import tensorflow as tf
import time
import threading
from skdata.mnist.views import OfficialVectorClassification
import numpy as np
import models
from tqdm import trange
from time import sleep

# load data entirely into memory 
data = OfficialVectorClassification()
trIdx = data.sel_idxs[:]
features = data.all_vectors[trIdx]
labels = data.all_labels[trIdx]

batch_size = 128
def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(features))
        np.random.shuffle(idxs)
        shuf_features = features[idxs]
        shuf_labels = labels[idxs]
        for batch_idx in range(0, len(features), batch_size):
            images_batch = shuf_features[batch_idx:batch_idx + batch_size] / 255.
            images_batch = images_batch.astype("float32")
            labels_batch = shuf_labels[batch_idx:batch_idx + batch_size]
            yield images_batch, labels_batch

class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self):
        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
        self.dataY = tf.placeholder(dtype=tf.int64, shape=[None, ])
        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        self.queue = tf.RandomShuffleQueue(shapes=[[28*28], []],
                                           dtypes=[tf.float32, tf.int64],
                                           capacity=2000,
                                           min_after_dequeue=1000)

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])

    def dequeue_batch(self,N):
        """
        Return's tensors containing a batch of images and labels
        """
        images_batch, labels_batch = self.queue.dequeue_many(N)
        return images_batch, labels_batch

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for dataX, dataY in data_iterator():
            sess.run(self.enqueue_op, feed_dict={self.dataX:dataX, self.dataY:dataY})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads

# Doing anything with data on the CPU is generally a good idea.
with tf.device("/cpu:0"):
    custom_runner = CustomRunner()
    images_batch, labels_batch = custom_runner.dequeue_batch(64)

model = models.LogisticRegression(images_batch)

logits,end_points = model.network()
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_batch)

# for monitoring
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
sess.run(tf.global_variables_initializer())

# start the tensorflow QueueRunner's
tf.train.start_queue_runners(sess=sess)
# start our custom queue runner's threads
custom_runner.start_threads(sess)

t = trange(100000, desc='Bar desc', leave=True)
for i in t:
    _, loss_val = sess.run([train_op, loss_mean])
    t.set_description('Loss (batch %f)' % loss_val)
