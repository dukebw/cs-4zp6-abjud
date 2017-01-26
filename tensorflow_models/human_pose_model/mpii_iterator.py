import timethis
import numpy as np
from scipy.io import loadmat
from mpii_read import parse_mpii_data_from_mat
from PIL import Image
import img_utils
import tensorflow as tf
import threading
import write_tf_record as mpii_utils
import cPickle as pickle

MPII_MAT_PATH = '/mnt/data/datasets/MPII_HumanPose/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
MPII_PCL_PATH = 'MPII_Dataset.p'
MPII_IMG_PATH = '/mnt/data/datasets/MPII_HumanPose/images/'
IS_TRAIN = 1

def read_mat():
    print('Reading the mpii .mat file')
    mpii_dataset_mat = loadmat(MPII_MAT_PATH, struct_as_record=False, squeeze_me=True)['RELEASE']
    return parse_mpii_data_from_mat(mpii_dataset_mat, MPII_IMG_PATH, is_train)

def read_pickle():
    return pickle.load(open('MPII_Dataset.p','rb'))

print 'reading pickle'
MPII_RAW = read_pickle()


class MPIIBatchIterator(object):
    '''
    batches up the mpii dataset
    '''
    def __init__(self, raw_data=MPII_RAW, BATCH_SIZE=1, WIDTH=380, HEIGHT=380, is_train=True):
        self.data = raw_data
        self._curr_step = 0
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.BATCH_SIZE = BATCH_SIZE
        self.coder = mpii_utils.ImageCoder(tf.Session())

    def __iter__(self):
        return self

    def _load_img(self, fname):
        '''
        Descr: Loads an image with a given size
        Load an image into PIL format.
        # Arguments
        path: path to image file
        grayscale: boolean
        target_size: None (default to original size)
            or (img_height, img_width)
        '''
        img = Image.open(fname)
        img = img.convert('RGB')
        img = img.resize((self.WIDTH, self.HEIGHT),Image.ANTIALIAS)
        img_arr = img_utils.img2array(img)
        return img_arr

    def _load_binary_heatmap(self, pose):
        '''
        Loads a person and gives a corresponding binary heatmap
        Args: person
        Returns: a binary map with a 3D array [NUM_JOINTS, WIDTH, HEIGHT]
        each binary map is an array of zeros everywhere except in a 5 pixel radius around the joint location
        '''
        # Do convnets understand the flattening and the deflattening operations well?
        x_sparse_joints, y_sparse_joints, sparse_joint_indices, is_visible_list = pose
        Ydet = np.zeros([16, self.WIDTH, self.HEIGHT])
        i = 0
        for (x,y) in zip(x_sparse_joints,y_sparse_joints):
            if is_visible_list[i] != 0:
                joint_pixel_x =int(self.WIDTH*x) + self.WIDTH/2
                joint_pixel_y =int(self.HEIGHT*y) + self.HEIGHT/2
                Ydet[i][joint_pixel_x:joint_pixel_x+5, joint_pixel_y:joint_pixel_y+5] = 1
            i+=1

        return Ydet

    def gaussian2d(self, mean_W, mean_H, sigma = 5.0/380.0):
        x = np.linspace(-0.5, 0.5, 380)
        z1 = np.exp(-(np.power(x - mean_H, 2.0)/(2.0 * np.power(sigma, 2.0))))
        z1 *= (1.0 / (sigma * np.sqrt(2.0 * 3.1415)))
        z2 = np.exp(-(np.power(x - mean_W, 2.0)/(2.0 * np.power(sigma, 2.0))))
        z2 *= (1.0 / (sigma * np.sqrt(2.0 * 3.1415)))
        z_2d = np.matmul(np.reshape(z1, [self.WIDTH, 1]), np.reshape(z2, [1, self.WIDTH]))
        return z_2d

    def _load_gaussian_heatmap(self, pose, img_tensor):
        '''
        Gaussian with a standard deviation of 5 pixels
        '''
        x_sparse_joints, y_sparse_joints, sparse_joint_indices, is_visible_list = pose
        Yreg = np.zeros([17, self.WIDTH, self.HEIGHT])
        Yreg[0] = img_tensor[0:self.WIDTH, 0:self.HEIGHT,0]
        i = 1
        for (x,y) in zip(x_sparse_joints, y_sparse_joints):
            if is_visible_list[i] != 0:
                Yreg[i] = self.gaussian2d(x,y)
        return Yreg


    def _fetch_example(self, fname, people_in_img):
        with tf.gfile.FastGFile(name=fname, mode='rb') as f:
            image_jpeg = f.read()
        img_shape = self.coder.decode_jpeg(image_jpeg)

        for person in people_in_img:
            person_rect = mpii_utils.find_person_bounding_box(person, img_shape)

            padded_img_dim, person_shape_xy, padding_xy = mpii_utils.find_padded_person_dim(person_rect)

            img_tensor = self.coder.crop_pad_resize(image_jpeg,
                                                         person_rect.top_left,
                                                         person_shape_xy,
                                                         padding_xy,
                                                         padded_img_dim)

            pose = mpii_utils.extract_labeled_joints(person.joints,
                                                     person_shape_xy,
                                                     padding_xy,
                                                     person_rect.top_left)

        return img_tensor, pose

    def next(self):
        '''
        gets the next set of imgs in memory
        '''
        # first get all filenames in a list
        fname_list = self.data.img_filenames[self._curr_step: self._curr_step + self.BATCH_SIZE]
        # Get a batch of people in imgs corresponding to the filenames
        people_in_imgs = self.data.people_in_imgs[self._curr_step: self._curr_step + self.BATCH_SIZE]
        # initialize input and target arrays
        X = np.zeros([self.BATCH_SIZE, self.WIDTH, self.HEIGHT, 3])
        # Ydet is a binary map for the whole image 
        Ydet = np.zeros([self.BATCH_SIZE, 16, self.WIDTH, self.HEIGHT]) 
        Yreg = np.zeros([self.BATCH_SIZE, 17, self.WIDTH, self.HEIGHT])
        # set the value of each batch element of the 4D array 
        for i in range(self.BATCH_SIZE):
                img_tensor, pose = self._fetch_example(fname_list[i],people_in_imgs[i])
                X[i] = img_tensor
                Ydet[i] = self._load_binary_heatmap(pose)
                Yreg[i] = self._load_gaussian_heatmap(pose, img_tensor)
        self._curr_step += self.BATCH_SIZE
        return X, Ydet, Yreg

    def pil_process_img(self, fname_list):
        img_list = []
        for fname in fname_list:
            img = self._load_img(fname)
            img_list.append(img)

        return img_list


class SimpleRunner(object):
    """
    This class manages the background threads needed to fill
        a queue full of data.
    """
    def __init__(self):
        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, 380, 380, 3])
        self.dataY = tf.placeholder(dtype=tf.float32, shape=[None, 380, 380, 3])
        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        self.queue = tf.RandomShuffleQueue(shapes=[[380,380,3], [380,380,3]],
                                           dtypes=[tf.float32, tf.float32],
                                           capacity=50,
                                           min_after_dequeue=15)

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        images_batch, labels_batch = self.queue.dequeue_many(20)
        return images_batch, labels_batch

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        batch_iterator = MPIIBatchIterator(MPII_RAW)
        for dataX, dataY in batch_iterator:
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
'''
# Doing anything with data on the CPU is generally a good idea.
with tf.device("/cpu:0"):
    simple_runner = SimpleRunner()
    images_batch, labels_batch = simple_runner.get_inputs()
'''
'''
# simple model
w = tf.get_variable("w1", [28*28, 10])
y_pred = tf.matmul(images_batch, w)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, labels_batch)

# for monitoring
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)
'''
'''sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
sess.run(tf.global_variables_initializer())

# start the tensorflow QueueRunner's
tf.train.start_queue_runners(sess=sess)
# start our custom queue runner's threads
simple_runner.start_threads(sess)

while True:
    image=sess.run(images_batch)
    print image
    '''
