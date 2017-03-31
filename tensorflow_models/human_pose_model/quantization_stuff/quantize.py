import tensorflow as tf
from tensorflow.python.framework import graph_io
from networks.vgg_bulat import vgg_16_bn_relu
RESTORE_PATH = '/export/mlrg/tjonsson/Winter2017/HumanPoseEstimation/mcmaster-text-to-motion-database/tensorflow_models/human_pose_model/quantization_stuff'
SAVE_DIR = '/export/mlrg/tjonsson/Winter2017/HumanPoseEstimation/mcmaster-text-to-motion-database/tensorflow_models/human_pose_model/quantization_stuff'
IMAGE_DIM = 380

# THE SEVEN STEPS
# 1. Define inference
# 2. Save checkpoint
# 3. Save graph_def
# 4. Match graph_def with checkpoints to generate a quantized output_graph
# 5. Use output graph in C++ file
# 6. Create a build file and build the file
# 6. Run executable on Jetson
# The inference function
def inference(image_bytes_feed, neural_network):
    """This function sets up a computation graph that will decode from JPEG the
    input placeholder image `image_bytes_feed`, pad and resize it to shape
    [IMAGE_DIM, IMAGE_DIM], then run joint inference on the image using the
    "Two VGG-16s cascade" model.
    """
    decoded_image = tf.image.decode_jpeg(contents=image_bytes_feed)
    decoded_image = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)

    image_shape = tf.shape(input=decoded_image)
    height = image_shape[0]
    width = image_shape[1]
    pad_dim = tf.cast(tf.maximum(height, width), tf.int32)
    offset_height = tf.cast(tf.cast((pad_dim - height), tf.float32)/2.0, tf.int32)
    offset_width = tf.cast(tf.cast((pad_dim - width), tf.float32)/2.0, tf.int32)
    padded_image = tf.image.pad_to_bounding_box(image=decoded_image,
                                                offset_height=offset_height,
                                                offset_width=offset_width,
                                                target_height=pad_dim,
                                                target_width=pad_dim)

    resized_image = tf.image.resize_images(images=padded_image,
                                           size=[IMAGE_DIM, IMAGE_DIM])

    normalized_image = tf.subtract(x=resized_image, y=0.5)
    normalized_image = tf.multiply(x=normalized_image, y=2.0)

    normalized_image = tf.reshape(tensor=normalized_image,
                                  shape=[IMAGE_DIM, IMAGE_DIM, 3])

    normalized_image = tf.expand_dims(input=normalized_image, axis=0)
    num_output_channels = 16
    logits, endpoints = neural_network(normalized_image,
                                       num_output_channels,
                                       is_detector_training=False,
                                       is_regressor_training=False)

    return logits, endpoints


# We need a function that uses graph_io to prepare our network for quantization
def prepare(input_graph_name, neural_network):
    # Define the graph
    with tf.Graph().as_default():
        image_bytes_feed = tf.placeholder(dtype=tf.string)
        logits,_ = inference(image_bytes_feed, neural_network)
        shape = logits.get_shape()
        merged_logits = tf.reshape(tf.reduce_max(logits, 3),[1, 380, 380, 1])
        pose = tf.cast(merged_logits, tf.float32)
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=RESTORE_PATH)
        assert latest_checkpoint is not None

        restorer = tf.train.Saver(var_list=tf.global_variables())
        restorer.restore(sess=session, save_path=latest_checkpoint)


def freeze_my_graph(sess, checkpoint_name, input_graph_name):
    sess = prepare(input_graph_name,vggbulatfn)
    graph_io.write_graph(sess.graph, SAVE_DIR, input_graph_name)
    # We save out the graph to disk, and then call the const conversion routine.
    checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, "saved_checkpoint")
    input_graph_path = os.path.join(FLAGS.model_dir, input_graph_name)
    input_saver_def_path = ""
    input_binary = False
    input_checkpoint_path = checkpoint_prefix + "-0"
    output_node_names = "Dense2/output_node"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(FLAGS.model_dir, output_graph_name)
    clear_devices = False
    freeze_graph.freeze_graph(input_graph_path,
                              input_saver_def_path,
                              input_binary,
                              input_checkpoint_path,
                              output_node_names,
                              restore_op_name,
                              filename_tensor_name,
                              output_graph_path,
                              clear_devices)

