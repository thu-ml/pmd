import tensorflow as tf
from tensorflow.python.training import optimizer
import numpy as np
from skimage import io, img_as_ubyte
from skimage.exposure import rescale_intensity
from six.moves import range
import os
import dataset
import time


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def save_image_collections(x, filename, shape=(10, 10), scale_each=False,
                           transpose=False):
    """
    :param shape: tuple
        The shape of final big images.
    :param x: uint8 numpy array
        Input image collections. (number_of_images, rows, columns, channels) or
        (number_of_images, channels, rows, columns)
    :param scale_each: bool
        If true, rescale intensity for each image.
    :param transpose: bool
        If true, transpose x to (number_of_images, rows, columns, channels),
        i.e., put channels behind.
    :return: uint8 numpy array
        The output image.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    n = x.shape[0]
    if transpose:
        x = x.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x[i] = rescale_intensity(x[i], out_range=(0, 1))
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print('Shape too small to contain all images')
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)


def reuse(scope):
    """
    A decorator for transparent reuse of tensorflow
    `Variables <https://www.tensorflow.org/api_docs/python/tf/Variable>`_ in a
    function. The decorated function will automatically create variables the
    first time they are called and reuse them thereafter.
    .. note::
        This decorator is internally implemented by tensorflow's
        :func:`make_template` function. See `its doc
        <https://www.tensorflow.org/api_docs/python/tf/make_template>_`
        for requirements on the target function.
    :param scope: A string. The scope name passed to tensorflow
        `variable_scope()
        <https://www.tensorflow.org/api_docs/python/tf/variable_scope>`_.
    """
    return lambda f: tf.make_template(scope, f)


def load_image_data(data, n_xl, n_channels, output_batch_size):
    if data == 'mnist':
        # Load MNIST
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'data', 'mnist.pkl.gz')
        x_train, t_train, x_valid, t_valid, _, _ = \
            dataset.load_mnist_realval(data_path)
        x_train = np.vstack([x_train, x_valid]).astype('float32')
        x_train = np.reshape(x_train, [-1, n_xl, n_xl, n_channels])

        x_train2 = x_train[:output_batch_size]
        t_train2 = t_train[:output_batch_size]
        t_train2 = np.nonzero(t_train2)[1]
        order = np.argsort(t_train2)
        sorted_x_train = x_train2[order]
    elif data == 'svhn':
        # Load SVHN data
        print('Reading svhn...')
        time_read = -time.time()
        print('Train')
        x_train = np.load('data/svhn_train1_x.npy')
        y_train = np.load('data/svhn_train1_y.npy')
        print('Test')
        x_test = np.load('data/svhn_test_x.npy')
        y_test = np.load('data/svhn_test_y.npy')
        time_read += time.time()
        print('Finished in {:.4f} seconds'.format(time_read))

        x_train2 = x_train[:output_batch_size]
        y_train2 = y_train[:output_batch_size]
        order = np.argsort(y_train2)
        sorted_x_train = x_train2[order]
    else:
        # Load LFW data
        print('Reading lfw...')
        time_read = -time.time()
        x_train = np.load('data/lfw.npy').astype(np.float32)
        print(x_train.shape)
        x_train = np.reshape(x_train, [-1, n_xl, n_xl, n_channels])
        time_read += time.time()
        print('Finished in {:.4f} seconds'.format(time_read))

        sorted_x_train = x_train[:output_batch_size]

    return x_train, sorted_x_train


class Batches:
    def __init__(self, X, batch_size):
        self.X     = X
        self.start = 0
        self.batch_size = batch_size

    def _call(self):
        ret = self.X[self.start:self.start+self.batch_size]
        self.start = min(self.start+self.batch_size, self.X.shape[0])
        if self.start == self.X.shape[0]:
            self.start = 0
        return ret
