#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from model import get_inception_score
from six.moves import range
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scipy.misc
import utils
from scipy.optimize import linear_sum_assignment
from ae import ConvAE
from utils import reuse, Batches
from generators import get_generator
from discriminators import get_discriminator
from pmd import PMD, PMDGAN

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'mnist', 'mnist, svhn or lfw')
tf.app.flags.DEFINE_string('dist', 'l1', 'Type of distance: l1, l2, cos or mmd')
tf.app.flags.DEFINE_string('arch', 'fc', 'Network architecture: fc, conv, ae or adv')
tf.app.flags.DEFINE_float('lr0', 3e-4, 'Maximal learning rate')
tf.app.flags.DEFINE_integer('t0', 10, 'Maximal learning rate')
tf.app.flags.DEFINE_integer('lag', 1, 'Number of epoches to output image')
tf.app.flags.DEFINE_integer('epoches', 100, 'Number of epoches')
tf.app.flags.DEFINE_integer('bs', 500, 'Batch size')
tf.app.flags.DEFINE_integer('mbs', 500, 'Match batch size')
tf.app.flags.DEFINE_string('match', 'r', 'Matching algorithm: e(exact), r(randomized) or s(sparse)')
tf.app.flags.DEFINE_float('bw', 1, 'Bandwidth for MMD, only useful for arch=ae')
tf.app.flags.DEFINE_float('reg', 10, 'regularization coefficient for WGAN-GP')

tf.app.flags.DEFINE_integer('n_z', 40, 'Number of latent dims')
tf.app.flags.DEFINE_integer('n_f', 128, 'Number of feature dims (only useful for arch=adv)')
tf.app.flags.DEFINE_integer('n_code', 40, 'Number of AE code dims')

tf.app.flags.DEFINE_integer('ae_bs', 100, 'Batch size for training autoencoder')
tf.app.flags.DEFINE_integer('ae_epoches', 100, 'Number of epoches')
tf.app.flags.DEFINE_float('ae_lr0', 3e-3, 'Maximal learning rate')
tf.app.flags.DEFINE_integer('ae_t0', 10, 'Maximal learning rate')

# Define model parameters
if FLAGS.dataset == 'mnist':
    n_xl       = 28
    n_channels = 1
    ngf        = 32
elif FLAGS.dataset == 'lfw':
    n_xl       = 32
    n_channels = 1
    ngf        = 64
else:
    n_xl       = 32
    n_channels = 3
    ngf        = 64

n_z    = FLAGS.n_z
n_f    = FLAGS.n_f
n_code = FLAGS.n_code
n_x    = n_xl * n_xl * n_channels
BaseModel = PMD if FLAGS.arch != 'adv' else PMDGAN

class MyPMD(BaseModel):
    def __init__(self, X, X_test, xshape, generator, run_name, ae=None, reg=None, F=None, D=None):
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
                                    generator, lambda x: x, reg, F, D)

    def _callback(self, sess, info):
        if info.epoch % FLAGS.lag != 0:
            return
        x_real = self.X_test
        x_gen  = self.generate1(sess, {self.batch_size_ph: FLAGS.mbs})
        if FLAGS.arch == 'ae':
            x_real = self.ae.decode(x_real, sess)
            x_gen  = self.ae.decode(x_gen,  sess)

        name = '{}/small_images_{}.jpg'.format(self.run_name, info.epoch)
        utils.save_image_collections(x_gen[np.random.permutation(FLAGS.mbs)[:50]], 
                                     name, scale_each=True, shape=(5, 10))
        print('Epoch {} (total {:.1f}, dist {:.1f}, match {:.1f}, sgd {:.1f} s): loss = {}'.format(info.epoch, info.time, info.time_gen, info.time_align, info.time_opt, info.loss))

        if FLAGS.arch == 'adv':
            print('AE loss = {}, GP = {}'.format(info.reg, info.gp))

	if FLAGS.dataset == 'cifar':
            images = []
            for i in range(600 if info.epoch%100==0 else 10):
                images.append(self.generate1(sess, {self.batch_size_ph: 100}))
            images = np.concatenate(images, axis=0)
            images = list((images*128+128).astype(np.int32))
            print('Inception score = {}'.format(get_inception_score(images)))


def main(argv=None):
    tf.set_random_seed(1237)
    np.random.seed(1237)

    # Load data
    x_train, sorted_x_train = \
            utils.load_image_data(FLAGS.dataset, n_xl, n_channels, FLAGS.mbs)
    xshape = (-1, n_xl, n_xl, n_channels)
    print('Data shape = {}'.format(x_train.shape))

    x_train        = x_train*2 - 1
    sorted_x_train = sorted_x_train*2 - 1

    # Make some data
    is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
    generator = get_generator(FLAGS.dataset, FLAGS.arch, 
                                           n_code if FLAGS.arch=='ae' else n_x, 
                                           n_xl, n_channels, n_z, ngf, is_training, 'transformation')
    if FLAGS.arch == 'adv':
        discriminator = get_discriminator(FLAGS.dataset, FLAGS.arch,
                                          n_x, n_xl, n_channels, n_f, ngf//2, is_training)
        decoder       = get_generator(FLAGS.dataset, FLAGS.arch, 
                                      n_x, n_xl, n_channels, n_f, ngf, is_training, 'decoder')

    # Define training/evaluation parameters
    run_name = 'results/{}_{}_{}_{}_c{}_mbs{}_bs{}_lr{}_t0{}'.format(
        FLAGS.dataset, FLAGS.arch, FLAGS.dist, FLAGS.match, n_code, 
        FLAGS.mbs, FLAGS.bs, FLAGS.lr0, FLAGS.t0)

    if not os.path.exists(run_name):
        os.makedirs(run_name)

    # Build the computation graph
    if FLAGS.arch == 'ae':
        ae = ConvAE(x_train, (None, n_xl, n_xl, n_channels), ngf)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ae.train(sess)
            x_code        = ae.encode(x_train, sess)
            sorted_x_code = ae.encode(sorted_x_train, sess)

        model = MyPMD(x_code, sorted_x_code, xshape,
                      generator, run_name, ae)
    elif FLAGS.arch == 'adv':
        model = MyPMD(x_train, sorted_x_train, xshape,
                      generator, run_name, reg=FLAGS.reg, F=discriminator, D=decoder)
    else:
        model = MyPMD(x_train, sorted_x_train, xshape,
                      generator, run_name)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.arch == 'ae':
            ae.train(sess)

        print('Training...')
        model.train(sess, gen_dict={model.batch_size_ph: FLAGS.mbs, is_training: False},
                          opt_dict={model.batch_size_ph: FLAGS.bs,  is_training: True},
                          iters=((x_train.shape[0]-1)//FLAGS.mbs)+1)

if __name__ == "__main__":
    tf.app.run()
