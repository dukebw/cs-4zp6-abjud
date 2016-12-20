
#from Hvass labs
# data input is from Imanol Schlag
# Here I just define all the value in the neural network

# placeholder variable is one that must be fed with data


width = 128
height = 128
img_size_flat = width*height
num_channels = 3
num_parts = 1


x = tf.placeholder(tf.float, shape=[None, img_size_flat], name='x')

x_image = tf.reshape(x, [-1, 128, 128, num_channels])
# data is initially a array
# reshape it as a tensor

# this is where I'm going to put the 64*64 sized output

outputSize = 64*64
y = tf.placeholder(tf.float, shape=[None, outputSize], name='y')
y_heatmap = tf.reshape(y, [-1, 64, 64, num_channels, num_parts])

# now define tensorflow variables -- a tensorFlow variable holds its value
# across calls to run

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape =[length]))

# here is where I should define the neural network
def new_conv_layer():
    return nothing

result = tf.subtract(y_heatmap, new_conv_layer(), name = None)
result = tf.square(result, name =None)
cost = tf.reduce_mean(result)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

session = tf.Session()
session.run(tf.initialize_all_variables())
train_batch_size = 20

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.






        # all I have to do is use my own thread to generate this
        # x_batch and y_true_batch

        
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


