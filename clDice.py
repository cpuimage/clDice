import tensorflow as tf


def dilation2d(x, kernel_size=3, dilations=1, strides=1):
    kernel = tf.zeros([kernel_size, kernel_size, x.get_shape().as_list()[-1]], tf.float32)
    y = tf.nn.dilation2d(x,
                         filters=kernel,
                         strides=[1, strides, strides, 1], padding="SAME", data_format="NHWC",
                         dilations=[1, dilations, dilations, 1])
    return y


def erosion2d(x, kernel_size=3, dilations=1, strides=1):
    kernel = tf.zeros([kernel_size, kernel_size, x.get_shape().as_list()[-1]], tf.float32)
    y = tf.nn.erosion2d(x,
                        filters=kernel,
                        strides=[1, strides, strides, 1], padding="SAME", data_format="NHWC",
                        dilations=[1, dilations, dilations, 1])
    return y


def fixed_soft_skeletonize(x, maximum_iterations=10, kernel_size=3, dilations=1):
    for _ in tf.range(0, maximum_iterations):
        min_pool_x = erosion2d(x, kernel_size=kernel_size, dilations=dilations)
        contour = tf.nn.relu(dilation2d(min_pool_x, kernel_size=kernel_size, dilations=dilations) - min_pool_x)
        x = tf.nn.relu(x - contour)
    return x


def soft_skeletonize(x, maximum_iterations=10, kernel_size=3, dilations=1, threshold=1.):
    def body(_, skelitonize):
        eroded = erosion2d(skelitonize, kernel_size=kernel_size, dilations=dilations)
        skelitonize = tf.nn.relu(
            skelitonize - tf.nn.relu(dilation2d(eroded, kernel_size=kernel_size, dilations=dilations) - eroded))
        return [eroded, skelitonize]

    def cond(prev_eroded, _):
        return tf.reduce_any(tf.reduce_sum(prev_eroded, (1, 2, 3)) > threshold)

    _, x = tf.while_loop(cond=cond,
                         body=body,
                         maximum_iterations=maximum_iterations,
                         loop_vars=[x, x])
    return x


def norm_intersection(center_line, vessel):
    intersection = tf.reduce_sum(center_line * vessel, axis=(1, 2, 3), keepdims=True)
    return tf.math.divide_no_nan(intersection, tf.reduce_sum(center_line, axis=(1, 2, 3), keepdims=True))


def soft_cldice_losses(y_true, y_pred, true_skeleton=None, maximum_iterations=10, fixed_iterations=True):
    if fixed_iterations:
        soft_skeletonize_func = fixed_soft_skeletonize
    else:
        soft_skeletonize_func = soft_skeletonize
    pred_skeleton = soft_skeletonize_func(y_pred, maximum_iterations=maximum_iterations)
    if true_skeleton is None:
        true_skeleton = soft_skeletonize_func(y_true, maximum_iterations=maximum_iterations)
    iflat = norm_intersection(pred_skeleton, y_true)
    tflat = norm_intersection(true_skeleton, y_pred)
    loss = 1. - tf.math.divide_no_nan(2. * iflat * tflat, iflat + tflat)
    return loss
