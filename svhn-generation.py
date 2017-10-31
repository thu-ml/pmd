#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zhusuan as zs
from ae import ConvAE

import dataset
import scipy.misc
import utils
from scipy.optimize import linear_sum_assignment
from assignments import lsa, approx_lsa, assignments, sparse_lsa
import distances


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dist', 'l1', 'Type of distance: l1, l2, cos or mmd')
tf.app.flags.DEFINE_string('arch', 'conv', 'Network architecture: fc, conv or ae')
tf.app.flags.DEFINE_integer('obs', 100, 'Optimize batch size')
tf.app.flags.DEFINE_float('lr0', 3e-4, 'Maximal learning rate')
tf.app.flags.DEFINE_integer('lag', 5, 'Number of epoches to output image')
tf.app.flags.DEFINE_integer('epoches', 100, 'Number of epoches')
tf.app.flags.DEFINE_integer('bs', 2000, 'Batch size')
tf.app.flags.DEFINE_string('match', 'r', 'Matching algorithm: e(exact), r(randomized) or s(sparse)')
tf.app.flags.DEFINE_integer('ae_bs', 100, 'Batch size for training autoencoder')
tf.app.flags.DEFINE_integer('ae_epoches', 100, 'Number of epoches')
tf.app.flags.DEFINE_float('ae_lr0', 3e-3, 'Maximal learning rate')
tf.app.flags.DEFINE_integer('ae_t0', 10, 'Maximal learning rate')
tf.app.flags.DEFINE_float('bw', 8, 'Bandwidth for MMD, only useful for arch=ae')
tf.app.flags.DEFINE_integer('n_code', 40, 'Code dimension')

# Define model parameters
n_xl = 32
n_channels = 3
n_z = 40
n_code = FLAGS.n_code
ngf = 64


class ConvAE:
    def __init__(self, x, n_h, ngf, normalizer_params):
        @zs.reuse('encoder')
        def encoder(x):
            h = layers.conv2d(x, ngf*2, 5, stride=2,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.conv2d(h, ngf*4, 5, stride=2,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.conv2d(h, ngf*8, 5, stride=2,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.flatten(h)
            h = layers.fully_connected(h, n_h, 
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            return h

        @zs.reuse('decoder')
        def decoder(z):
            h = layers.fully_connected(z, num_outputs=ngf*8*4*4,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)
            h = tf.reshape(h, [-1, 4, 4, ngf*8])
            h = layers.conv2d_transpose(h, ngf*4, 5, stride=2,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.conv2d_transpose(h, ngf*2, 5, stride=2,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            x = layers.conv2d_transpose(h, 3, 5, stride=2, activation_fn=tf.nn.sigmoid)
            return x

        self.encoder = encoder
        self.decoder = decoder

        code = encoder(x)
        self.recons = decoder(code)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.recons - x), [1, 2, 3]))

        self.encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        self.all_encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        self.all_decoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')

    def filters(self):
        with tf.variable_scope('decoder/Conv2d_transpose_2', reuse=True):
            return tf.get_variable('weights')


def flatten(x):
    return np.reshape(x, [x.shape[0], -1])


if FLAGS.arch == 'fc':
    @zs.reuse('generator')
    def generator(z_ph, n_x, normalizer_params):
        h = layers.fully_connected(z_ph, 500, 
                normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
        h = layers.fully_connected(h, 500, 
                normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
        x = layers.fully_connected(h, n_x, activation_fn=tf.nn.sigmoid)
        return tf.reshape(x, [-1, n_xl, n_xl, n_channels])
elif FLAGS.arch == 'conv':
    @zs.reuse('generator')
    def generator(z_ph, n_x, normalizer_params):
        h = layers.fully_connected(z_ph, num_outputs=ngf*8*4*4,
                                   normalizer_fn=layers.batch_norm,
                                   normalizer_params=normalizer_params)
        h = tf.reshape(h, [-1, 4, 4, ngf*8])
        h = layers.conv2d_transpose(h, ngf*4, 5, stride=2,
                normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
        h = layers.conv2d_transpose(h, ngf*2, 5, stride=2,
                normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
        x = layers.conv2d_transpose(h, 3, 5, stride=2, activation_fn=tf.nn.sigmoid)
        return x
else:
    @zs.reuse('generator')
    def generator(z_ph, n_x, normalizer_params):
        h = layers.fully_connected(z_ph, 500, 
                normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
        h = layers.fully_connected(h, 500, 
                normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
        x = layers.fully_connected(h, n_x)
        return x


def main(argv=None):
    tf.set_random_seed(1237)
    dist = FLAGS.dist

    # Load SVHN data
    print('Reading svhn...')
    time_read = -time.time()
    print('Train')
    x_train = np.load('svhn_train1_x.npy')
    y_train = np.load('svhn_train1_y.npy')
    print('Test')
    x_test = np.load('svhn_test_x.npy')
    y_test = np.load('svhn_test_y.npy')
    time_read += time.time()
    print('Finished in {:.4f} seconds'.format(time_read))

    np.random.seed(1234)
    n_x = n_xl * n_xl * n_channels
    n = x_train.shape[0]
    x_code = np.zeros((x_train.shape[0], n_code))

    output_batch_size = FLAGS.bs
    Fy = 80
    Fx = output_batch_size * 2 // Fy

    # Make some data
    x_train2 = x_train[:output_batch_size]
    y_train2 = y_train[:output_batch_size]
    order = np.argsort(y_train2)
    sorted_x_train = x_train2[order]
    sorted_x_code = np.zeros((sorted_x_train.shape[0], n_code))

    # Define training/evaluation parameters
    match_batch_size = FLAGS.bs
    epoches = FLAGS.epoches
    burnin_epoches = 10
    optimize_batch_size = FLAGS.obs
    if FLAGS.dist == 'mmd':
        optimize_batch_size = match_batch_size
    learning_rate = FLAGS.lr0
    t0 = 10
    n_ix = 20
    n_iy = 20

    run_name = 'results/svhnn_{}_{}_{}_c{}_m{}_o{}_lr{}_t0{}_s'.format(
        FLAGS.arch, dist, FLAGS.match, n_code, match_batch_size, optimize_batch_size, learning_rate, t0)
    ae_name = 'results/ae_svhn_c{}_o{}_lr{}_t0{}'.format(n_code, FLAGS.ae_bs, FLAGS.ae_lr0, FLAGS.ae_t0)

    if not os.path.exists(run_name):
        os.mkdir(run_name)
    if not os.path.exists(ae_name):
        os.mkdir(ae_name)

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    ae_is_training = tf.placeholder(tf.bool, shape=[], name='ae_is_training')
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None,
                         'decay': 0.9}
    ae_normalizer_params = {'is_training': ae_is_training,
                            'updates_collections': None}
    batch_size_ph = tf.placeholder(tf.int32, shape=())

    x_ph = tf.placeholder(tf.float32, shape=(None, n_xl, n_xl, n_channels))
    code_ph = tf.placeholder(tf.float32, shape=(None, n_code))
    z_op = tf.random_uniform(shape=(batch_size_ph, n_z), minval=-1, maxval=1)
    z_ph = tf.placeholder(tf.float32, shape=(None, n_z))

    # GMN
    if FLAGS.arch != 'ae':
        generated_image = generator(z_ph, n_x, normalizer_params)
        flattened_x = layers.flatten(x_ph)
        flattened_g = layers.flatten(generated_image)
    else:
        ae = ConvAE(x_ph, n_code, ngf, ae_normalizer_params)
        x_code_op = ae.encoder(x_ph)
        generated_code = generator(z_ph, n_code, normalizer_params)
        generated_image = ae.decoder(generated_code)
        flattened_x = code_ph
        flattened_g = generated_code
        ae_filters = ae.filters()


    def my_distance(x, y):
        # The matched objective (L1 distance)
        if dist == 'l1':
            matched_obj = distances.l1(x, y)
        elif dist == 'l2':
            matched_obj = distances.l2(x, y)
        else:
            if FLAGS.arch != 'ae': # TODO
                matched_obj = tf.sqrt(distances.mmd(x, y, 1) + 
                                      distances.mmd(x, y, 1.5) + 
                                      distances.mmd(x, y, 2.5) + 
                                      distances.mmd(x, y, 3.2) + 
                                      distances.mmd(x, y, 4.5) + 
                                      distances.mmd(x, y, 6) + 
                                      distances.mmd(x, y, 10) +
                                      distances.mmd(x, y, 20)) / 8
            else:
                matched_obj = tf.sqrt(distances.mmd(x, y, FLAGS.bw))
        return matched_obj

    my_pw_distance = lambda x, y: distances.pw_l2(x, y) if dist == 'l2' else distances.pw_l1(x, y)
    matched_obj = my_distance(flattened_x, flattened_g)
    gen_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope='generator')


    print('Generator variables:')
    for i in gen_var_list:
        print(i.name, i.get_shape())
    if FLAGS.arch == 'ae':
        print('Encoder variables:')
        for i in ae.encoder_vars:
            print(i.name, i.get_shape())
        print('Decoder variables:')
        for i in ae.decoder_vars:
            print(i.name, i.get_shape())

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    infer = optimizer.minimize(matched_obj, var_list=gen_var_list)
    if FLAGS.arch == 'ae':
        train_ae = optimizer.minimize(ae.loss, var_list=ae.encoder_vars+ae.decoder_vars)
        saver = tf.train.Saver(max_to_keep=1, var_list=ae.all_encoder_vars+ae.all_decoder_vars)

    def generate_imgs(run_name, epoch):
        z = sess.run(z_op, feed_dict={batch_size_ph: output_batch_size})
        if FLAGS.arch != 'ae':
            g_img = sess.run(generated_image, feed_dict={z_ph: z, is_training: False})
            obj_matrix = my_pw_distance(flatten(sorted_x_train), flatten(g_img))
        else:
            g_code, g_img = sess.run([generated_code, generated_image],
                    feed_dict={z_ph: z, is_training: False, ae_is_training: False})
            obj_matrix = my_pw_distance(sorted_x_code, g_code)

        rs, cs = lsa(obj_matrix)
        assigned_mx = g_img[assignments(rs, cs)]
        match_result = obj_matrix[rs, cs].mean()
        if FLAGS.match == 'r':
            rs2, cs2 = approx_lsa(obj_matrix)
        elif FLAGS.match == 's':
            rs2, cs2 = sparse_lsa(obj_matrix)
        else:
            rs2, cs2 = rs, cs
        match_result2 = obj_matrix[rs2, cs2].mean()

        # Interweave the imgs
        all_imgs = np.reshape(np.hstack((sorted_x_train, assigned_mx)), (-1, n_xl, n_xl, n_channels))
        name = '{}/outfile_{}_{}.jpg'.format(run_name, epoch, match_result)
        utils.save_image_collections(all_imgs, name, scale_each=True, shape=(Fx, Fy))

        name = '{}/images_{}_{}.jpg'.format(run_name, epoch, match_result)
        utils.save_image_collections(g_img.reshape(-1, n_xl, n_xl, n_channels), name, scale_each=True, shape=(Fx, Fy//2))

        name = '{}/small_images_{}_{}.jpg'.format(run_name, epoch, match_result)
        utils.save_image_collections(g_img[:50].reshape(-1, n_xl, n_xl, n_channels), name, scale_each=True, shape=(5, 10))

        # Make interpolation
        for run in range(10):
            pts = 5
            zs = sess.run(z_op, feed_dict={batch_size_ph: pts})
            izs = np.zeros((pts * 20, n_z)).astype(np.float32)
            cnt = 0
            for i in range(pts):
                s = zs[i]
                t = zs[(i+1)%pts]
                delta = (t-s) / 20
                for j in range(20):
                    izs[cnt] = s + delta*j
                    cnt += 1

            g_img = sess.run(generated_image, feed_dict={z_ph: izs, is_training: False, ae_is_training: False})
            name = '{}/interpolation{}_{}_run{}.jpg'.format(run_name, epoch, match_result, run)
            utils.save_image_collections(g_img.reshape(-1, n_xl, n_xl, n_channels), name, scale_each=True, shape=(pts, 20))

        return match_result, match_result2

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.arch == 'ae':
            # Train autoencoder
            ckpt_file = tf.train.latest_checkpoint(ae_name)
            if ckpt_file is not None:
                print('Restoring autoencoder...')
                saver.restore(sess, ckpt_file)
            else:
                print('Training autoencoder...')
                for epoch in range(1, FLAGS.ae_epoches+1):
                    time_epoch = -time.time()
                    np.random.shuffle(x_train)
                    losses = []
                    for t in range(x_train.shape[0] // FLAGS.ae_bs):
                        x_batch = x_train[t*FLAGS.ae_bs : (t+1)*FLAGS.ae_bs]
                        _, l = sess.run([train_ae, ae.loss],
                                feed_dict={x_ph: x_batch,
                                           learning_rate_ph: 
                                               FLAGS.ae_lr0 * FLAGS.ae_t0 / (FLAGS.ae_t0 + epoch),
                                           ae_is_training: True})
                        losses.append(l)
                    time_epoch += time.time()
                    avg_loss = np.mean(losses)
                    print('Epoch {} ({:.1f})s: Reconstruction loss = {}'.format(epoch, time_epoch, avg_loss))

                save_path = os.path.join(ae_name, 'ae.ckpt')
                saver.save(sess, save_path)

            print('Visualizing autoencoder')
            # Visualize AE filters
            f = sess.run(ae_filters, feed_dict={ae_is_training: False})
            f = np.transpose(f, [3, 0, 1, 2])
            name = '{}/filter.png'.format(ae_name)
            utils.save_image_collections(f, name, scale_each=True, shape=(16, 16))

            # Visualize AE reconstruction
            r = sess.run(ae.recons, feed_dict={x_ph: sorted_x_train, ae_is_training: False})
            imgs = np.hstack((sorted_x_train, r))
            imgs = np.reshape(imgs, [-1, n_xl, n_xl, n_channels])
            name = '{}/recons.png'.format(ae_name)
            utils.save_image_collections(imgs, name, scale_each=True, shape=(Fx, Fy))

            # Compute the code for x_train and sorted_x_train
            def compute_code(x_train):
                x_code = np.zeros((x_train.shape[0], n_code)).astype(np.float32)
                for t in range(x_train.shape[0] // FLAGS.ae_bs + 1):
                    x_batch = x_train[t*FLAGS.ae_bs : (t+1)*FLAGS.ae_bs]
                    if x_batch.shape[0] == 0:
                        continue
                    c = sess.run(x_code_op, 
                            feed_dict={x_ph: x_batch, ae_is_training: False})
                    x_code[t*FLAGS.ae_bs : (t+1)*FLAGS.ae_bs] = c

                return x_code

            print('Computing code')
            x_code = compute_code(x_train)
            sorted_x_code = compute_code(sorted_x_train)

        print('Training...')
        # Train
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            time_dist = 0
            time_match = 0
            time_sgd = 0

            perm = np.random.permutation(n)
            x_train = x_train[perm]
            x_code = x_code[perm]

            ws = []
            losses = []
            for t in range(n // match_batch_size):
                x_batch = x_train[t*match_batch_size : (t+1)*match_batch_size]
                code_batch = x_code[t*match_batch_size : (t+1)*match_batch_size]

                # Generate latent variable
                z = sess.run(z_op, feed_dict={batch_size_ph: match_batch_size})

                # Compute distance
                time_dist -= time.time()
                if FLAGS.arch != 'ae':
                    g_img = sess.run(generated_image, 
                                     feed_dict={z_ph: z, is_training: False, ae_is_training: False})
                    obj_matrix = my_pw_distance(flatten(g_img), flatten(x_batch))
                else:
                    if dist != 'mmd':
                        g_code = sess.run(generated_code,
                                         feed_dict={z_ph: z, is_training: False, ae_is_training: False})
                        obj_matrix = my_pw_distance(g_code, code_batch)
                time_dist += time.time()

                # Compute minimal weight matching
                time_match -= time.time()
                if dist == 'mmd':
                    ws = 0
                    assigned_x = x_batch
                    assigned_code = code_batch
                    match_result = 0
                else:
                    if FLAGS.match == 'e':
                        rs, cs = lsa(obj_matrix)
                    elif FLAGS.match == 'r':
                        rs, cs = approx_lsa(obj_matrix)
                    else:
                        rs, cs = sparse_lsa(obj_matrix)
                    match_result = obj_matrix[rs, cs].mean()
                    ws.append(match_result)
                    assigned_x = x_batch[assignments(rs, cs)]
                    assigned_code = code_batch[assignments(rs, cs)]
                time_match += time.time()
                # print(match_result, time_match)

                # Perform SGD
                time_sgd -= time.time()
                for sgd_iters in range(match_batch_size // optimize_batch_size):
                    sgd_z_batch = z[sgd_iters * optimize_batch_size : (sgd_iters+1) * optimize_batch_size]
                    sgd_x_batch = assigned_x[sgd_iters * optimize_batch_size : (sgd_iters+1) * optimize_batch_size]
                    sgd_code_batch = assigned_code[sgd_iters * optimize_batch_size : (sgd_iters+1) * optimize_batch_size]
                    # Gradient descent
                    _, l = sess.run([infer, matched_obj], 
                                    feed_dict={z_ph: sgd_z_batch,
                                               x_ph: sgd_x_batch,
                                               code_ph: sgd_code_batch,
                                               batch_size_ph: optimize_batch_size,
                                               learning_rate_ph: learning_rate * t0 / (t0 + epoch),
                                               is_training: True, ae_is_training: False})
                    losses.append(l)
                time_sgd += time.time()

            time_epoch += time.time()

            # Generate figures
            if epoch % FLAGS.lag == 0:
                match_result, match_result2 = generate_imgs(run_name, epoch)
                print('Epoch {} (total {:.1f}, dist {:.1f}, match {:.1f}, sgd {:.1f} s): W distance = {} approx W distance = {}, loss = {}'.format(epoch, time_epoch, time_dist, time_match, time_sgd, match_result, match_result2, np.mean(losses)))


if __name__ == "__main__":
    tf.app.run()
