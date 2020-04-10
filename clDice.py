import tensorflow as tf
import numpy as np


def dilation2d(x, kernel_size=3, dilations=1, strides=1):
    weight = 1. / (kernel_size * kernel_size)
    kernel = np.asarray(np.ones([kernel_size, kernel_size]), np.float32)
    kernel = kernel.reshape(list(kernel.shape) + [1, 1])
    kernel = np.tile(kernel, [1, 1, np.shape(x)[-1], 1]) * weight
    y = tf.nn.dilation2d(x,
                         filters=tf.constant(np.squeeze(kernel, -1) - weight, tf.float32),
                         strides=[1, strides, strides, 1], padding="SAME", data_format="NHWC",
                         dilations=[1, dilations, dilations, 1])
    return y


def erosion2d(x, kernel_size=3, dilations=1, strides=1):
    weight = 1. / (kernel_size * kernel_size)
    kernel = np.asarray(np.ones([kernel_size, kernel_size]), np.float32)
    kernel = kernel.reshape(list(kernel.shape) + [1, 1])
    kernel = np.tile(kernel, [1, 1, np.shape(x)[-1], 1]) * weight
    y = tf.nn.erosion2d(x,
                        filters=tf.constant(np.squeeze(kernel, -1) - weight, tf.float32),
                        strides=[1, strides, strides, 1], padding="SAME", data_format="NHWC",
                        dilations=[1, dilations, dilations, 1])
    return y


def soft_skeletonize(x, thresh_width=10, kernel_size=3, dilations=1):
    for _ in tf.range(0, thresh_width):
        erosion_out = erosion2d(x, kernel_size=kernel_size, dilations=dilations)
        contour = tf.nn.relu(dilation2d(erosion_out, kernel_size=kernel_size, dilations=dilations) - erosion_out)
        x = tf.nn.relu(x - contour)
    return x


def norm_intersection(center_line, vessel):
    intersection = tf.reduce_sum(center_line * vessel, axis=(1, 2, 3), keepdims=True)
    return tf.math.divide_no_nan(intersection, tf.reduce_sum(center_line, axis=(1, 2, 3), keepdims=True))


def soft_cldice_loss(y_true, y_pred, true_skeleton=None, thresh_width=10):
    pred_skeleton = soft_skeletonize(y_pred, thresh_width=thresh_width)
    if true_skeleton is None:
        true_skeleton = soft_skeletonize(y_true, thresh_width=thresh_width)
    iflat = norm_intersection(pred_skeleton, y_true)
    tflat = norm_intersection(true_skeleton, y_pred)
    loss = 1. - tf.math.divide_no_nan(2. * iflat * tflat, iflat + tflat)
    return loss
