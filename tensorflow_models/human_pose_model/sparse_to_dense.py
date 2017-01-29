import tensorflow as tf

def _sparse_joints_to_dense_one_dim(dense_shape, joint_indices, joints, num_joints):
    """Converts a sparse vector of joints in a single dimension to dense
    joints, and returns those dense joints.
    """
    sparse_joints = tf.sparse_merge(sp_ids=joint_indices,
                                    sp_values=joints,
                                    vocab_size=num_joints)
    dense_joints = tf.sparse_tensor_to_dense(sp_input=sparse_joints,
                                             default_value=0)

    return tf.reshape(tensor=dense_joints, shape=dense_shape), sparse_joints


def _sparse_joints_to_dense_inner(dense_shape,
                                  x_joints,
                                  y_joints,
                                  joint_indices,
                                  num_joints):
    """
    """
    x_dense_joints, x_sparse_joints = _sparse_joints_to_dense_one_dim(
        dense_shape,
        joint_indices,
        x_joints,
        num_joints)

    y_dense_joints, _ = _sparse_joints_to_dense_one_dim(
        dense_shape,
        joint_indices,
        y_joints,
        num_joints)

    weights = tf.sparse_to_dense(sparse_indices=x_sparse_joints.indices,
                                 output_shape=dense_shape,
                                 sparse_values=1,
                                 default_value=0)

    return x_dense_joints, y_dense_joints, weights


def sparse_joints_to_dense_single_example(x_joints,
                                          y_joints,
                                          joint_indices,
                                          num_joints):
    """
    """
    return _sparse_joints_to_dense_inner([num_joints],
                                         x_joints,
                                         y_joints,
                                         joint_indices,
                                         num_joints)


def sparse_joints_to_dense(training_batch, num_joints):
    """Converts a sparse vector of joints to a dense format, and also returns a
    set of weights indicating which joints are present.

    Args:
        training_batch: A batch of training images with associated joint
            vectors.

    Returns:
        (dense_joints, weights) tuple, where dense_joints is a dense vector of
        shape [batch_size, NUM_JOINTS], with zeros in the indices not
        present in the sparse vector. `weights` contains 1s for all the present
        joints and 0s otherwise.
    """
    x_dense_joints, y_dense_joints, weights = _sparse_joints_to_dense_inner(
        [training_batch.batch_size, num_joints],
        training_batch.x_joints,
        training_batch.y_joints,
        training_batch.joint_indices,
        num_joints)

    return x_dense_joints, y_dense_joints, tf.concat(concat_dim=1, values=[weights, weights])
