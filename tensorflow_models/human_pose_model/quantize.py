import tensorflow as tf
from tensorflow.python.framework import graph_io
from networks.vgg_bulat import vgg_16_bn_relu

SAVE_DIR = '.'
IMAGE_DIM = 256


def get_joint_position_inference_graph(image_bytes_feed):
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

    logits, _ = two_vgg_16s_cascade(normalized_image, 16, False, False)

    return logits


# We need a function that uses graph_io to prepare our network for quantization
def prepare(input_graph_name, get_logits_function)
    # Define the graph
    with tf.Graph().as_default():
        image_bytes_feed = tf.placeholder(dtype=tf.string)
        logits = get_joint_position_inference_graph(image_bytes_feed)
        shape = logits.get_shape()
        merged_logits = tf.reshape(tf.reduce_max(logits, 3),[1, 380, 380, 1])
        pose = tf.cast(merged_logits, tf.float32)
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=RESTORE_PATH)
        assert latest_checkpoint is not None

        restorer = tf.train.Saver(var_list=tf.global_variables())
        restorer.restore(sess=session, save_path=latest_checkpoint)
    # Write graph in protobuf format
    graph_io.write_graph(sess.graph, SAVE_DIR, input_graph_name)


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_graph_path, clear_devices, "")
