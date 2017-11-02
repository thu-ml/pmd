import tensorflow as tf
from tensorflow.contrib import layers
from assignments import get_assignments
from distances   import my_distance, pw_l1, pw_l2
import numpy as np
import time

FLAGS = tf.app.flags.FLAGS

class PMDInfo(object):
    pass

class GAN(object):
    def __init__(self, noise1, noise2, ns1, ns2, T1, T2, reg, F):
        self.noise1 = noise1
        self.noise2 = noise2
        self.T1     = T1
        self.T2     = T2
        self.F      = F
        self.reg    = reg

        self.noise_ph_1 = tf.placeholder(tf.float32, shape=ns1)
        self.noise_ph_2 = tf.placeholder(tf.float32, shape=ns2)
        self.X1 = T1(self.noise_ph_1)
        self.X2 = T2(self.noise_ph_2)

        self.F1, self.F2 = F(self.X1), F(self.X2)

        self.matched_obj = tf.reduce_mean(self.F1 - self.F2)

        # Output variables
        trans_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='transformation')
        feat_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='discriminator')
        print('Transformation variables:')
        for i in trans_vars:
            print(i.name, i.get_shape())
        print('Critic variables:')
        for i in feat_vars:
            print(i.name, i.get_shape())

        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate_ph, decay=0.5)
        self.train_op  = self.optimizer.minimize(self.matched_obj,  var_list=trans_vars)
        self.feat_op   = self.optimizer.minimize(-self.matched_obj, var_list=feat_vars)
        self.clip_op   = tf.group(*[var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var
                                    in feat_vars])

    def _generate(self, sess, dict, noise1=None, noise2=None):
        Z1, Z2    = self._generate_noise(sess, dict, noise1, noise2)
        feed_dict = {self.noise_ph_1: Z1,
                     self.noise_ph_2: Z2}
        feed_dict.update(dict)
        X1, X2, F1, F2 = sess.run([self.X1, self.X2, self.F1, self.F2], 
                                  feed_dict=feed_dict)
        return Z1, Z2, X1, X2, F1, F2

    def _generate_noise(self, sess, dict, noise1=None, noise2=None):
        n2n = noise2 is not None
        if noise1 is None:
            noise1 = self.noise1
        if noise2 is None:
            noise2 = self.noise2

        Z1 = noise1() if callable(noise1) else sess.run(noise1, feed_dict=dict)
        Z2 = noise2() if callable(noise2) else sess.run(noise2, feed_dict=dict)
        return Z1, Z2

    def _align(self, F1, F2):
        flatten    = lambda x: np.reshape(x, [x.shape[0], -1])
        obj_matrix = self.my_pw_distance(flatten(F1), flatten(F2))

        if FLAGS.dist == 'mmd':
            match_result = 0
            a = np.arange(F1.shape[0])
        else:
            a, match_result = get_assignments(obj_matrix, FLAGS.match)

        return a, match_result

    def train(self, sess, gen_dict, opt_dict, iters):
        for epoch in range(1, FLAGS.epoches+1):
            losses = []
            learning_rate = FLAGS.lr0 * FLAGS.t0 / (FLAGS.t0 + epoch)

            info = PMDInfo()
            info.time_gen   = 0
            info.time_align = 0
            info.time_opt   = 0

            cnt = 0
            for it in range(iters):
                t = time.time()
                Z1, Z2 = self._generate_noise(sess, gen_dict)
                info.time_gen += time.time() - t
                if Z1.shape[0] != Z2.shape[0]:
                    continue

                t0 = time.time()
                f    = {self.noise_ph_1: Z1, self.noise_ph_2: Z2,
                        self.learning_rate_ph: learning_rate}
                f.update(opt_dict)

                cnt += 1
                if cnt % 6 == 0:
                    _, l = sess.run([self.train_op, self.matched_obj], feed_dict=f)
                    #print('Gen ', l)
                    losses.append(l)
                else:
                    _, l = sess.run([self.feat_op,  self.matched_obj], feed_dict=f)
                    #print('Cri ', l)

                info.time_opt += time.time() - t0
                info.time      = info.time_gen + info.time_align + info.time_opt
                info.epoch     = epoch
                info.loss      = np.mean(losses)

            self._callback(sess, info)

    def _callback(self):
        pass

