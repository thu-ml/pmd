import tensorflow as tf
from tensorflow.contrib import layers
from assignments import get_assignments
from distances   import my_distance, pw_l1, pw_l2
import numpy as np
import time

FLAGS = tf.app.flags.FLAGS

class PMDInfo(object):
    pass

class PMDGAN(object):
    def __init__(self, noise1, noise2, ns1, ns2, T1, T2, reg, F, D):
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
        self.R1, self.R2 = D(self.F1), D(self.F2)

        self.matched_obj = my_distance(FLAGS.dist, FLAGS.arch, FLAGS.bw, 
                                       layers.flatten(self.F1), layers.flatten(self.F2))
        self.my_pw_distance = lambda x, y: (pw_l2(x, y) 
                  if FLAGS.dist == 'l2' else pw_l1(x, y))
        l2_loss      = lambda x: tf.reduce_mean(tf.square(x))
        self.r_loss  = 8 * (l2_loss(layers.flatten(self.R1 - self.X1)) + 
                            l2_loss(layers.flatten(self.R2 - self.X2)))

        alpha        = tf.random_uniform(tf.stack([tf.shape(self.X1)[0], 1, 1, 1]),
                                         minval=0., maxval=1.)
        differences  = self.X2 - self.X1
        interpolates = self.X1 + differences * alpha
        gradients    = layers.flatten(tf.gradients(self.F(interpolates), [interpolates])[0])
        slopes       = tf.sqrt(tf.reduce_sum(tf.square(gradients), [-1]))
        self.gp      = reg * tf.reduce_mean(tf.square(slopes-1))

        disc_cost    = -self.matched_obj + self.r_loss + self.gp

        # Output variables
        trans_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='transformation')
        disc_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope='discriminator') 
        dec_vars   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='decoder') 
        print('Transformation variables:')
        for i in trans_vars:
            print(i.name, i.get_shape())
        print('Disc variables:')
        for i in disc_vars:
            print(i.name, i.get_shape())
        print('Decoder variables:')
        for i in dec_vars:
            print(i.name, i.get_shape())

        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
        #self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate_ph)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph, beta1=0.5, beta2=0.9)
        self.train_op  = self.optimizer.minimize(self.matched_obj, var_list=trans_vars)
        self.feat_op = self.optimizer.minimize(disc_cost, var_list=disc_vars+dec_vars)
        self.clip_op = tf.group(*[var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var
                                  in disc_vars])
        self.disc_vars = disc_vars


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
        X1, X2, F1, F2 = sess.run([self.X1, self.X2, self.F1, self.F2], 
                                  feed_dict=feed_dict)
        return Z1, Z2, X1, X2, F1, F2

    def _align(self, F1, F2):
        flatten    = lambda x: np.reshape(x, [x.shape[0], -1])
        obj_matrix = self.my_pw_distance(flatten(F1), flatten(F2))

        if FLAGS.dist == 'mmd':
            match_result = 0
            a = np.arange(F1.shape[0])
        else:
            a, match_result = get_assignments(obj_matrix, FLAGS.match)

        return a, match_result

    def _generate1(self, sess, dict):
        Z1 = self.noise1() if callable(self.noise1) else sess.run(self.noise1, feed_dict=dict)

        feed_dict = {self.noise_ph_1: Z1}
        feed_dict.update(dict)
        X1 = sess.run(self.X1, feed_dict=feed_dict)
        return X1

    def train(self, sess, gen_dict, opt_dict, iters):
        for epoch in range(1, FLAGS.epoches+1):
            losses = []
            regs   = []
            gps    = []
            learning_rate = FLAGS.lr0 * FLAGS.t0 / (FLAGS.t0 + epoch)

            info = PMDInfo()
            info.time_gen   = 0
            info.time_align = 0
            info.time_opt   = 0

            interval = 101 if epoch<=4 else 6

            cnt = 0
            for it in range(iters):
                t = time.time()
                Z1, Z2, X1, X2, F1, F2 = self._generate(sess, gen_dict)
                info.time_gen += time.time() - t
                if Z1.shape[0] != Z2.shape[0]:
                    continue

                t = time.time()
                a, w           = self._align(F1, F2)
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

                    if cnt % interval == 0 or self.reg is None:
                        _, l = sess.run([self.train_op, self.matched_obj], feed_dict=f)
                        losses.append(l)
#print(l)
                    else:
                        _, l, reg, gp = sess.run([self.feat_op, self.matched_obj, self.r_loss, self.gp], 
                                             feed_dict=f)
                        regs.append(reg)
                        gps.append(gp)
                info.time_opt += time.time() - t0

            #exit(0)
            info.time      = info.time_gen + info.time_align + info.time_opt
            info.epoch     = epoch
            info.loss      = np.mean(losses)
            info.reg       = np.mean(regs)
            info.gp        = np.mean(gps)

            self._callback(sess, info)

    def _callback(self):
        pass

