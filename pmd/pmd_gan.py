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
        l2_loss      = lambda x: tf.reduce_mean(tf.square(x))
        self.r_loss  = 8 * (l2_loss(layers.flatten(self.R1 - self.X1)) + 
                            l2_loss(layers.flatten(self.R2 - self.X2)))
        disc_cost    = -self.matched_obj #+ self.r_loss

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
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate_ph)
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

        return Z1, Z2


    def train(self, sess, gen_dict, opt_dict, iters):
        for epoch in range(1, FLAGS.epoches+1):
            losses = []
            regs   = []
            learning_rate = FLAGS.lr0 * FLAGS.t0 / (FLAGS.t0 + epoch)

            info = PMDInfo()
            info.time_gen   = 0
            info.time_align = 0
            info.time_opt   = 0

            interval = 101 if epoch<=4 else 6

            cnt = 0
            for it in range(iters):
                t = time.time()
                Z1, Z2 = self._generate(sess, gen_dict)
                info.time_gen += time.time() - t
                if Z1.shape[0] != Z2.shape[0]:
                    continue

                t0 = time.time()
                cnt += 1
                f    = {self.noise_ph_1: Z1, self.noise_ph_2: Z2,
                        self.learning_rate_ph: learning_rate}
                f.update(opt_dict)

                sess.run(self.clip_op)

                if cnt % interval == 0 or self.reg is None:
                    _, l = sess.run([self.train_op, self.matched_obj], feed_dict=f)
                    print(l)
                    losses.append(l)

                    f1, f2 = sess.run([self.F1, self.F2], feed_dict=f)
                    print(np.square(f1-f2).sum(1))

                    #for var in self.disc_vars:
                    #    print(var.name, sess.run(var))
                else:
                    _, l, reg = sess.run([self.feat_op, self.matched_obj, self.r_loss], 
                                         feed_dict=f)
                    regs.append(reg)
                    #print(l, reg)
                info.time_opt += time.time() - t0

            #exit(0)
            info.time      = info.time_gen + info.time_align + info.time_opt
            info.epoch     = epoch
            info.loss      = np.mean(losses)
            info.reg       = np.mean(regs)

            self._callback(sess, info)

    def _callback(self):
        pass

