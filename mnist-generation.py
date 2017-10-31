#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scipy.misc
import utils
from scipy.optimize import linear_sum_assignment
from assignments import lsa, approx_lsa, assignments, sparse_lsa, get_assignments
import distances
from ae import ConvAE
from utils import reuse
from generators import get_generator
from new_pmd import PMD

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'mnist', 'mnist, svhn or lfw')
tf.app.flags.DEFINE_string('dist', 'l1', 'Type of distance: l1, l2, cos or mmd')
tf.app.flags.DEFINE_string('arch', 'fc', 'Network architecture: fc, conv or ae')
tf.app.flags.DEFINE_integer('obs', 100, 'Optimize batch size')
tf.app.flags.DEFINE_float('lr0', 3e-4, 'Maximal learning rate')
tf.app.flags.DEFINE_integer('t0', 10, 'Maximal learning rate')
tf.app.flags.DEFINE_integer('lag', 1, 'Number of epoches to output image')
tf.app.flags.DEFINE_integer('epoches', 100, 'Number of epoches')
tf.app.flags.DEFINE_integer('bs', 500, 'Batch size')
tf.app.flags.DEFINE_integer('mbs', 500, 'Match batch size')
tf.app.flags.DEFINE_string('match', 'r', 'Matching algorithm: e(exact), r(randomized) or s(sparse)')
tf.app.flags.DEFINE_float('bw', 1, 'Bandwidth for MMD, only useful for arch=ae')

tf.app.flags.DEFINE_integer('n_code', 40, 'Code dimension')
tf.app.flags.DEFINE_integer('ae_bs', 100, 'Batch size for training autoencoder')
tf.app.flags.DEFINE_integer('ae_epoches', 100, 'Number of epoches')
tf.app.flags.DEFINE_float('ae_lr0', 3e-3, 'Maximal learning rate')
tf.app.flags.DEFINE_integer('ae_t0', 10, 'Maximal learning rate')

# Define model parameters
if FLAGS.dataset == 'mnist':
    n_xl       = 28
    n_channels = 1
    ngf        = 32
else:
    n_xl       = 32
    n_channels = 3
    ngf        = 64

n_z    = 40
n_code = FLAGS.n_code
n_x    = n_xl * n_xl * n_channels
n_ix   = 20
n_iy   = 20
Fy     = 80
Fx     = FLAGS.mbs * 2 // Fy


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


class MyPMD(PMD):
    def __init__(self, X, X_test, xshape, generator, run_name, ae=None):
        self.X      = X
        self.X_test = X_test
        self.xshape = xshape
        self.run_name = run_name
        # Noise1 is a uniform noise
        self.batch_size_ph = tf.placeholder(tf.int32, shape=())
        self.noise1 = tf.random_uniform(shape=(self.batch_size_ph, n_z), minval=-1, maxval=1)
        # Noise2 is a batch generator
        batches = Batches(X, FLAGS.mbs)
        self.noise2 = lambda: batches._call()
        self.ae = ae

        ns2    = list(X.shape)
        ns2[0] = None
        super(MyPMD, self).__init__(self.noise1, self.noise2, 
                                    (None, n_z), ns2, 
                                    generator, lambda x: x)

    def _callback(self, sess, epoch, loss):
        if epoch % FLAGS.lag != 0:
            return
        _, _, x_gen, x_real = self._generate(sess, {self.batch_size_ph: FLAGS.mbs},
                                             noise2=lambda: self.X_test)
        x_gen, match_result = self._align(x_real, x_gen)
        if FLAGS.arch == 'ae':
            x_real = self.ae.decode(x_real, sess)
            x_gen  = self.ae.decode(x_gen,  sess)

        # Interweave the imgs
        all_imgs = np.reshape(np.hstack((x_real, x_gen)), self.xshape)
        name = '{}/outfile_{}_{}.jpg'.format(self.run_name, epoch, match_result)
        utils.save_image_collections(all_imgs,   name, scale_each=True, shape=(Fx, Fy))

        name = '{}/images_{}_{}.jpg'.format(self.run_name, epoch, match_result)
        utils.save_image_collections(x_gen,      name, scale_each=True, shape=(Fx, Fy//2))

        name = '{}/small_images_{}_{}.jpg'.format(self.run_name, epoch, match_result)
        utils.save_image_collections(x_gen[:50], name, scale_each=True, shape=(5, 10))

        print('Epoch {} (total {:.1f}, dist {:.1f}, match {:.1f}, sgd {:.1f} s): approx W distance = {}, loss = {}'.format(epoch, 0, 0, 0, 0, match_result, loss))


def main(argv=None):
    tf.set_random_seed(1237)
    np.random.seed(1237)
    dist                = FLAGS.dist
    epoches             = FLAGS.epoches
    output_batch_size   = FLAGS.bs
    match_batch_size    = FLAGS.bs
    optimize_batch_size = match_batch_size if FLAGS.dist=='mmd' else FLAGS.obs
    learning_rate       = FLAGS.lr0

    # Load data
    x_train, sorted_x_train = \
            utils.load_image_data(FLAGS.dataset, n_xl, n_channels, output_batch_size)

    xshape    = list(x_train.shape)
    xshape[0] = -1
    n         = x_train.shape[0]

    # Make some data
    generator = get_generator(FLAGS.arch, n_x, n_xl, n_channels, ngf)

    # Define training/evaluation parameters
    run_name = 'results/{}_{}_{}_{}_c{}_m{}_o{}_lr{}_t0{}_s'.format(
        FLAGS.dataset, FLAGS.arch, dist, FLAGS.match, n_code, 
        match_batch_size, optimize_batch_size, learning_rate, FLAGS.t0)

    if not os.path.exists(run_name):
        os.mkdir(run_name)

    # Build the computation graph
    is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None,
                         'decay': 0.9}

    if FLAGS.arch == 'ae':
        ae = ConvAE(x_train, (None, n_xl, n_xl, n_channels), ngf)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ae.train(sess)
            x_code = ae.encode(x_train, sess)
            sorted_x_code = ae.encode(sorted_x_train, sess)

        model = MyPMD(x_code, sorted_x_code, xshape,
                      lambda x: generator(x, n_code, normalizer_params), run_name, ae)
    else:
        model = MyPMD(x_train, sorted_x_train, xshape,
                      lambda x: generator(x, n_x, normalizer_params), run_name)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.arch == 'ae':
            ae.train(sess)

        print('Training...')
        model.train(sess, gen_dict={model.batch_size_ph: FLAGS.mbs, is_training: False},
                          opt_dict={model.batch_size_ph: FLAGS.bs,  is_training: True},
                    iters=n//FLAGS.mbs)

if __name__ == "__main__":
    tf.app.run()
