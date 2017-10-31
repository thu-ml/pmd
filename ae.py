import tensorflow as tf
import os
from tensorflow.contrib import layers
from utils import reuse
import time
import numpy as np
import utils

FLAGS = tf.app.flags.FLAGS

class ConvAE:
    def __init__(self, X, x_shape, ngf):
        self.X       = X
        self.x_shape = x_shape
        self.ngf     = ngf

        with tf.variable_scope('ae'):
            self.build()

        self.ae_name  = 'results/ae_{}_c{}_o{}_lr{}_t0{}'.format(
                   FLAGS.dataset, FLAGS.n_code, FLAGS.ae_bs, FLAGS.ae_lr0, FLAGS.ae_t0)
        if not os.path.exists(self.ae_name):
            os.mkdir(self.ae_name)


    def build(self):
        ngf               = self.ngf
        self.x            = tf.placeholder(tf.float32, shape=self.x_shape)
        self.z            = tf.placeholder(tf.float32, shape=(None, FLAGS.n_code))
        self.is_training  = tf.placeholder(tf.bool, shape=[], name='is_training')
        normalizer_params = {'is_training': self.is_training,
                             'updates_collections': None}
 
        @reuse('encoder')
        def encoder(x):
            print('Encoder ', x.get_shape())
            h = layers.conv2d(x, ngf, 5, stride=2, 
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            print(h.get_shape())
            h = layers.conv2d(h, ngf*2, 5, stride=2,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            print(h.get_shape())
            h = layers.conv2d(h, ngf*4, 5, padding='VALID',
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            print(h.get_shape())
            h = layers.conv2d(h, FLAGS.n_code, 3, padding='VALID',
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            print(h.get_shape())
            return layers.flatten(h)

        @reuse('decoder')
        def decoder(z):
            h = tf.reshape(z, [-1, 1, 1, FLAGS.n_code])
            print('Decoder ', h.get_shape())
            h = layers.conv2d_transpose(h, ngf*4, 3, padding='VALID',
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            print(h.get_shape())
            h = layers.conv2d_transpose(h, ngf*2, 5, padding='VALID',
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            print(h.get_shape())
            h = layers.conv2d_transpose(h, ngf, 5, stride=2,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            print(h.get_shape())
            h = layers.conv2d_transpose(h, 1, 5, stride=2,
                    activation_fn=tf.nn.sigmoid)
            print('Output: ', h.get_shape())
            return h

        self.encoder = encoder
        self.decoder = decoder

        self.code = encoder(self.x)
        self.recons = decoder(self.code)
        self.decode_op = decoder(self.z)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.recons - self.x), [1, 2, 3]))

        self.encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ae/encoder')
        self.decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ae/decoder')
        self.all_encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ae/encoder')
        self.all_decoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ae/decoder')

        with tf.variable_scope('decoder/Conv2d_transpose_3', reuse=True):
            self.filters = tf.get_variable('weights')

        print('Encoder variables:')
        for i in self.encoder_vars:
            print(i.name, i.get_shape())
        print('Decoder variables:')
        for i in self.decoder_vars:
            print(i.name, i.get_shape())

        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.train_ae  = self.optimizer.minimize(self.loss, var_list=self.encoder_vars+self.decoder_vars)
        self.saver     = tf.train.Saver(max_to_keep=1, var_list=self.all_encoder_vars+self.all_decoder_vars)

    def train(self, sess):
        learning_rate_ph = self.learning_rate_ph
        optimizer        = self.optimizer
        train_ae         = self.train_ae
        saver            = self.saver

        # Train autoencoder
        ckpt_file = tf.train.latest_checkpoint(self.ae_name)
        if ckpt_file is not None:
            print('Restoring autoencoder...')
            saver.restore(sess, ckpt_file)
        else:
            print('Training autoencoder...')
            for epoch in range(1, FLAGS.ae_epoches+1):
                time_epoch = -time.time()
                np.random.shuffle(self.X)
                losses = []
                for t in range(self.X.shape[0] // FLAGS.ae_bs):
                    x_batch = self.X[t*FLAGS.ae_bs : (t+1)*FLAGS.ae_bs]
                    _, l = sess.run([train_ae, self.loss],
                            feed_dict={self.x: x_batch,
                                       learning_rate_ph: 
                                           FLAGS.ae_lr0 * FLAGS.ae_t0 / (FLAGS.ae_t0 + epoch),
                                       self.is_training: True})
                    losses.append(l)
                time_epoch += time.time()
                avg_loss = np.mean(losses)
                print('Epoch {} ({:.1f})s: Reconstruction loss = {}'.format(epoch, time_epoch, avg_loss))

            save_path = os.path.join(self.ae_name, 'ae.ckpt')
            saver.save(sess, save_path)

        print('Visualizing autoencoder')
        # Visualize AE filters
        f = sess.run(self.filters)
        f = np.transpose(f, [3, 0, 1, 2])
        name = '{}/filter.png'.format(self.ae_name)
        utils.save_image_collections(f, name, scale_each=True, shape=(16, 16))

        ## Visualize AE reconstruction
        #r = sess.run(self.recons, feed_dict={x_ph: sorted_x_train, ae_is_training: False})
        #imgs = np.hstack((sorted_x_train, r))
        #imgs = np.reshape(imgs, [-1, n_xl, n_xl, n_channels])
        #name = '{}/recons.png'.format(ae_name)
        #utils.save_image_collections(imgs, name, scale_each=True, shape=(Fx, Fy))

    def encode(self, X, sess):
        """Output the code of X"""
        x_code = np.zeros((X.shape[0], FLAGS.n_code), dtype=np.float32)
        for t in range(X.shape[0] // FLAGS.ae_bs + 1):
            x_batch = X[t*FLAGS.ae_bs : (t+1)*FLAGS.ae_bs]
            if x_batch.shape[0] == 0:
                continue
            c = sess.run(self.code, 
                    feed_dict={self.x: x_batch, self.is_training: False})
            x_code[t*FLAGS.ae_bs : (t+1)*FLAGS.ae_bs] = c

        return x_code
    
    def decode(self, Z, sess):
        """Output the decode of Z"""
        X = []
        for t in range(Z.shape[0] // FLAGS.ae_bs + 1):
            z_batch = Z[t*FLAGS.ae_bs : (t+1)*FLAGS.ae_bs]
            if z_batch.shape[0] == 0:
                continue
            x = sess.run(self.decode_op, 
                    feed_dict={self.z: z_batch, self.is_training: False})
            X.append(x)

        return np.concatenate(X, axis=0)
