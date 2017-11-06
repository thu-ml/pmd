import tensorflow as tf
from tensorflow.contrib import layers
from assignments import get_assignments
from distances   import my_distance, pw_l1, pw_l2
import numpy as np
import time

FLAGS = tf.app.flags.FLAGS

class PMDInfo(object):
    pass

class PMD(object):
    def __init__(self, noise1, noise2, ns1, ns2, T1, T2, F=None, D=None):
        self.noise1 = noise1
        self.noise2 = noise2
        self.T1     = T1
        self.T2     = T2
        self.F      = F

        self.noise_ph_1 = tf.placeholder(tf.float32, shape=ns1)
        self.noise_ph_2 = tf.placeholder(tf.float32, shape=ns2)
        self.X1 = T1(self.noise_ph_1)
        self.X2 = T2(self.noise_ph_2)

        self.matched_obj = my_distance(FLAGS.dist, FLAGS.arch, FLAGS.bw, 
                                       layers.flatten(self.X1), layers.flatten(self.X2))
        self.my_pw_distance = lambda x, y: (pw_l2(x, y) 
                  if FLAGS.dist == 'l2' else pw_l1(x, y))

        # Output variables
        trans_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='transformation')
        print('Transformation variables:')
        for i in trans_vars:
            print(i.name, i.get_shape())

        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.train_op  = self.optimizer.minimize(self.matched_obj, var_list=trans_vars)


    def _generate(self, sess, dict):
        Z1 = self.noise1() if callable(self.noise1) else sess.run(self.noise1, feed_dict=dict)
        Z2 = self.noise2() if callable(self.noise2) else sess.run(self.noise2, feed_dict=dict)

        feed_dict = {self.noise_ph_1: Z1,
                     self.noise_ph_2: Z2}
        feed_dict.update(dict)
        X1, X2 = sess.run([self.X1, self.X2], feed_dict=feed_dict)
        return Z1, Z2, X1, X2


    def generate1(self, sess, dict):
        Z1 = self.noise1() if callable(self.noise1) else sess.run(self.noise1, feed_dict=dict)

        feed_dict = {self.noise_ph_1: Z1}
        feed_dict.update(dict)
        X1 = sess.run(self.X1, feed_dict=feed_dict)
        return X1


    def _align(self, X1, X2):
        if FLAGS.dist == 'mmd':
            match_result = 0
            a = np.arange(X1.shape[0])
        else:
            flatten    = lambda x: np.reshape(x, [x.shape[0], -1])
            obj_matrix = self.my_pw_distance(flatten(X1), flatten(X2))
            a, match_result = get_assignments(obj_matrix, FLAGS.match)

        return a, match_result

    def train(self, sess, gen_dict, opt_dict, iters):
        for epoch in range(1, FLAGS.epoches+1):
            losses = []
            regs   = []
            learning_rate = FLAGS.lr0 * FLAGS.t0 / (FLAGS.t0 + epoch)

            info = PMDInfo()
            info.time_gen   = 0
            info.time_align = 0
            info.time_opt   = 0

            cnt = 0
            for it in range(iters):
                t = time.time()
                Z1, Z2, X1, X2 = self._generate(sess, gen_dict)
                info.time_gen += time.time() - t
                if Z1.shape[0] != Z2.shape[0]:
                    continue

                t = time.time()
                a, w           = self._align(X1, X2)
                Z2             = Z2[a]
                info.time_align += time.time() - t

                t0 = time.time()
                for t in range(FLAGS.mbs // FLAGS.bs):
                    cnt += 1
                    z1   = Z1[t*FLAGS.bs : (t+1)*FLAGS.bs]
                    z2   = Z2[t*FLAGS.bs : (t+1)*FLAGS.bs]
                    f    = {self.noise_ph_1: z1, self.noise_ph_2: z2,
                            self.learning_rate_ph: learning_rate}
                    f.update(opt_dict)

                    _, l = sess.run([self.train_op, self.matched_obj], feed_dict=f)
                    losses.append(l)

                info.time_opt += time.time() - t0

            info.time      = info.time_gen + info.time_align + info.time_opt
            info.epoch     = epoch
            info.loss      = np.mean(losses)

            self._callback(sess, info)

    def _callback(self):
        pass

