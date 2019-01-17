import tensorflow as tf


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _pairwise_distances(embeddings):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    return distances


def get_margin_loss(labels, features, num_classes, alpha=0.1, training=True):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])

    centers_batch = tf.gather(centers, labels)

    center_dist = tf.reduce_sum(tf.abs(tf.subtract(features, centers_batch)), axis=1)

    diff = centers_batch - features
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_time = tf.gather(unique_count, unique_idx)
    appear_time = tf.reshape(appear_time, [-1, 1])

    diff = diff / tf.cast((1+appear_time), tf.float32)
    diff = alpha * diff

    centers_update_up = tf.scatter_sub(centers, labels, diff)

    feature_center_pair_dist = tf.reduce_sum(tf.abs(tf.subtract(tf.expand_dims(features, 1), tf.expand_dims(centers, 0))), axis=2)
    
    feature_center_dist = tf.subtract(tf.expand_dims(center_dist, 1),  feature_center_pair_dist)


    feature_center_labels_equal = tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(tf.constant(list(range(num_classes)), dtype=tf.int64), 0))
    mask_feature_center = tf.to_float(tf.logical_not(feature_center_labels_equal))


    margin_loss = tf.reduce_sum(tf.nn.softplus(feature_center_dist)*mask_feature_center) / tf.reduce_sum(mask_feature_center)
 
    return margin_loss, centers, centers_update_up
	
	
def get_coral_loss(source, target):
    d = source.get_shape().as_list()[-1]
    # source covariance
    xm = tf.reduce_mean(source,axis=0,keep_dims=True) - source
    xc = tf.matmul(tf.transpose(xm), xm)
    # target covariance
    xmt = tf.reduce_mean(target, axis=0, keep_dims=True) - target
    xct = tf.matmul(tf.transpose(xmt), xmt)
    # frobenius norm between source and target
    loss = tf.reduce_mean(tf.abs((xc-xct)))
    return loss

	
def get_mmd_loss(source, target):
    d = source.get_shape().as_list()[-1]
    xm = tf.reduce_mean(source, axis=0)
    xmt = tf.reduce_mean(target, axis=0)
    loss = tf.reduce_mean(tf.abs((xm - xmt)))
    return loss