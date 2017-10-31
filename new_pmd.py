import tensorflow as tf
from tensorflow.contrib import layers
from assignments import get_assignments
from distances   import my_distance, pw_l1, pw_l2
import numpy as np

FLAGS = tf.app.flags.FLAGS

class PMD(object):
    def __init__(self, noise1, noise2, ns1, ns2, T1, T2):
        self.noise1 = noise1
        self.noise2 = noise2
        self.T1     = T1
        self.T2     = T2

        self.noise_ph_1 = tf.placeholder(tf.float32, shape=ns1)
        self.noise_ph_2 = tf.placeholder(tf.float32, shape=ns2)
        self.X1 = T1(self.noise_ph_1)
        self.X2 = T2(self.noise_ph_2)

        self.matched_obj = my_distance(FLAGS.dist, FLAGS.arch, FLAGS.bw, 
                                       layers.flatten(self.X1), layers.flatten(self.X2))
        self.my_pw_distance = lambda x, y: (pw_l2(x, y) 
                  if FLAGS.dist == 'l2' else pw_l1(x, y))

        self.outbs  = FLAGS.bs
        self.mbs    = FLAGS.bs
        self.optbs  = FLAGS.obs

        # Output variables
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='transformation')
        print('Transformation variables:')
        for i in var_list:
            print(i.name, i.get_shape())

        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.train_op  = self.optimizer.minimize(self.matched_obj, var_list=var_list)

    def _generate(self, sess, dict, noise1=None, noise2=None):
        n2n = noise2 is not None
        if noise1 is None:
            noise1 = self.noise1
        if noise2 is None:
            noise2 = self.noise2

        Z1 = noise1() if callable(noise1) else sess.run(noise1, feed_dict=dict)
        Z2 = noise2() if callable(noise2) else sess.run(noise2, feed_dict=dict)

        feed_dict = {self.noise_ph_1: Z1,
                     self.noise_ph_2: Z2}
        feed_dict.update(dict)
        X1, X2 = sess.run([self.X1, self.X2], 
                          feed_dict=feed_dict)
        return Z1, Z2, X1, X2

    def _align(self, X1, X2):
        flatten    = lambda x: np.reshape(x, [x.shape[0], -1])
        obj_matrix = self.my_pw_distance(flatten(X1), flatten(X2))

        if FLAGS.dist == 'mmd':
            match_result = 0
            assigned_x = X2
        else:
            a, match_result = get_assignments(obj_matrix, FLAGS.match)
            assigned_x = X2[a]

        return assigned_x, match_result

    def train(self, sess, gen_dict, opt_dict, iters):
        for epoch in range(1, FLAGS.epoches+1):
            losses = []
            learning_rate = FLAGS.lr0 * FLAGS.t0 / (FLAGS.t0 + epoch)

            for _ in range(iters):
                Z1, Z2, X1, X2 = self._generate(sess, gen_dict)
                X2, w          = self._align(X1, X2)

                for t in range(self.mbs // self.optbs):
                    z1   = Z1[t*self.optbs : (t+1)*self.optbs]
                    z2   = Z2[t*self.optbs : (t+1)*self.optbs]
                    f    = {self.noise_ph_1: z1, self.noise_ph_2: z2,
                            self.learning_rate_ph: learning_rate}
                    f.update(opt_dict)

                    _, l = sess.run([self.train_op, self.matched_obj], feed_dict=f)
                    losses.append(l)

            self._callback(sess, epoch, np.mean(losses))

    def _callback(self):
        pass

