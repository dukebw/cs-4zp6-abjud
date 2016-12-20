from utils import *

def build_graph()
    # Here is where we build our graph
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)
        keep_prob = tf.placeholder(tf.float32)

        trn_images_batch, trn_labels_batch = input_pipeline(FLAGS.train_record,
                                FLAGS.batch_size,FLAGS.num_epochs)

        skip_layers = ['fc6_W','fc6_b','fc7_W','fc7_b','fc8_W','fc8_b']
        # load pre-trained weights into orig. VGG16 network but skipping weights
        # for weights in skip_layer (i.e. top layers)
        N = 2 #Number of networks in cascade
        for i in xrange(N):
            vgg[i] = vgg16(trn_images_batch, keep_prob, 'vgg16_weights.npz', skip_layers)
            prediction[i] = vgg.probs
            pose_vector[i] = NormalizeOp(prediction[i])
        # Define loss and optimizer
        loss = tf.nn.softmax_cross_entropy_with_logits(prediction, trn_lbl_one_hot)

        # for monitoring
        loss_mean = tf.reduce_mean(loss)

        # get variables of top layers to train during fine-tuning
        trainable_layers = ["fc3","fc2","fc1"]
        train_vars = []
        for idx in trainable_layers:
            train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,idx)

        #train only the top layers of the VGG16 Net by passing var_list to the train_op
        train_op = tf.train.AdamOptimizer(0.0001).minimize(loss,var_list =train_vars)

        valid_prediction = vgg.probs
        test_prediction =  vgg.probs


        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())


        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        correct_prediction = tf.equal(tf.argmax(trn_lbl_one_hot,1), tf.argmax(prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.initialize_all_variables()
        init_locals = tf.initialize_local_variables()
