import numpy as np
import tensorflow as tf
from approxla import pairwise_l1

sess = tf.Session()

def l1(X1, X2):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(X1 - X2), -1))


def l2(X1, X2):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(X1 - X2), -1)))


def cos(X1, X2):
    inner_prod = tf.reduce_sum(X1 * X2, -1)
    xnorm = tf.sqrt(tf.reduce_sum(X1 * X1, -1))
    ynorm = tf.sqrt(tf.reduce_sum(X2 * X2, 1))
    return tf.reduce_mean(inner_prod / xnorm / ynorm)


def pw_l1(X, Y):
    D = np.zeros((X.shape[0], Y.shape[0]), np.float32)
    pairwise_l1(X, Y, D)
    return D


def pairwise_euclidean_distance(X, Y):
    # X: n * d
    # Y: m * d
    # Output: n * m matrix
    xy = tf.matmul(X, tf.transpose(Y))
    xx = tf.reduce_sum(tf.square(X), -1)
    yy = tf.reduce_sum(tf.square(Y), -1)
    D = tf.sqrt(tf.expand_dims(xx, 1) + tf.expand_dims(yy, 0) - 2 * xy)
    return D


def pairwise_cosine_similarity(X, Y):
    xy = tf.matmul(X, tf.transpose(Y))
    xx = tf.reduce_sum(tf.square(X), -1)
    yy = tf.reduce_sum(tf.square(Y), -1)
    D = xy / tf.expand_dims(tf.sqrt(xx), 1) / tf.expand_dims(tf.sqrt(yy), 0) 
    return D


def pairwise_rbf_kernel(X, Y, bandwidth):
    xy = tf.matmul(X, tf.transpose(Y))
    xx = tf.reduce_sum(tf.square(X), -1)
    yy = tf.reduce_sum(tf.square(Y), -1)
    D = tf.expand_dims(xx, 1) + tf.expand_dims(yy, 0) - 2 * xy
    K = tf.exp(-D / (2 * bandwidth**2))
    return K


def mmd(X1, X2, bandwidth):
    exx = tf.reduce_mean(pairwise_rbf_kernel(X1, X1, bandwidth))
    exy = tf.reduce_mean(pairwise_rbf_kernel(X1, X2, bandwidth))
    eyy = tf.reduce_mean(pairwise_rbf_kernel(X2, X2, bandwidth))
    return exx - 2*exy + eyy


def mmd_cond(X, Xneg, Y, bandwidth, size):
    exx = pairwise_rbf_kernel(X, Xneg, bandwidth)
    exy = pairwise_rbf_kernel(X, Y, bandwidth)
    return (exx - 2*exy) / size**2


def cmd(X1, X2, K):
    mean1 = tf.reduce_mean(X1, 0)
    mean2 = tf.reduce_mean(X2, 0)
    dist = tf.sqrt(tf.reduce_sum(tf.square(mean1 - mean2)))
    n_X1 = X1 - tf.expand_dims(mean1, 0)
    n_X2 = X2 - tf.expand_dims(mean2, 0)
    for i in range(2, K+1):
        m1 = tf.reduce_mean(tf.pow(n_X1, i), 0)
        m2 = tf.reduce_mean(tf.pow(n_X2, i), 0)
        dist += tf.sqrt(tf.reduce_sum(tf.square(m1 - m2)))
    return dist


def pw_l2(X, Y):
    return sess.run(l2_d, feed_dict={x: X, y: Y})


def pw_cos(X, Y):
    return sess.run(cos_d, feed_dict={x: X, y: Y})


def my_distance(dist, arch, bw, x, y):
    # The matched objective (L1 distance)
    if dist == 'l1':
        matched_obj = l1(x, y)
    elif dist == 'l2':
        matched_obj = l2(x, y)
    elif dist == 'l1m':
        matched_obj = tf.reduce_mean(tf.abs(x - y))
    elif dist == 'l1sqrt':
        matched_obj = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.abs(x-y), -1)))
    elif dist == 'wgan':
        matched_obj = tf.reduce_mean(x-y)
    else:
        if arch == 'adv':
            matched_obj = tf.sqrt(mmd(x, y, 1) + 
                                  mmd(x, y, 2) +
                                  mmd(x, y, 4) + 
                                  mmd(x, y, 8) + 
                                  mmd(x, y, 16))
        elif arch != 'ae':
            matched_obj = tf.sqrt(mmd(x, y, 4) + 
                                  mmd(x, y, 10) +
                                  mmd(x, y, 20) + 
                                  mmd(x, y, 40) + 
                                  mmd(x, y, 80) + 
                                  mmd(x, y, 160))
        else:
            matched_obj = tf.sqrt(mmd(x, y, bw))
    return matched_obj


x = tf.placeholder(tf.float32, [None, None])
y = tf.placeholder(tf.float32, [None, None])
l2_d = pairwise_euclidean_distance(x, y)
cos_d = pairwise_cosine_similarity(x, y)

