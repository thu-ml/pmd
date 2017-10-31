import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from assignments import lsa, approx_lsa, assignments, reverse
import distances
import time
import os, sys
from utils import reuse


class PMD:
    def __init__(self, params, conv=False):
        self.g = tf.Graph()

        self.params = params

        # Model parameters
        n_x = params.get('n_x', None)
        n_h = params.get('n_h', None)
        num_classes = params.get('num_classes', None)
        self.num_classes = num_classes
        reg = params.get('reg', None)
        keep_prob = params.get('keep_prob', None)
        self.reg = reg

        # Learning parameters
        self.learning_rate_0 = params.get('learning_rate_0', None)
        self.t0 = params.get('t0', None)
        self.epoches = params.get('epoches', None)
        self.batch_size = params.get('batch_size', None)
        self.dist = params.get('dist', None)
        self.bandwidth = params.get('bandwidth', None)
        self.bn = params.get('bn', None)        
        self.match = params.get('match', None)

        with self.g.as_default():
            self.y_ph = tf.placeholder(tf.int32, shape=[None])
            self.is_training = tf.placeholder(tf.bool, shape=[])
            self.learning_rate_ph = tf.placeholder(tf.float32, shape=[])
            normalizer_params = {'is_training': self.is_training,
                                 'updates_collections': None,
                                 'decay': 0.5}
            if not conv:
                self.x_ph_1 = tf.placeholder(tf.float32, shape=[None, n_x])
                self.x_ph_2 = tf.placeholder(tf.float32, shape=[None, n_x])

                # Layers
                if self.bn:
                    h1 = layers.fully_connected(self.x_ph_1, n_h, 
                                                activation_fn=tf.nn.sigmoid, 
                                                scope='nn',
                                                normalizer_fn=layers.batch_norm,
                                                normalizer_params=normalizer_params)
                    h2 = layers.fully_connected(self.x_ph_2, n_h, 
                                                activation_fn=tf.nn.sigmoid, 
                                                reuse=True, scope='nn',
                                                normalizer_fn=layers.batch_norm,
                                                normalizer_params=normalizer_params)
                else:
                    # h1 = layers.dropout(self.x_ph_1, keep_prob=0.8, is_training=self.is_training)
                    # h2 = layers.dropout(self.x_ph_2, keep_prob=0.8, is_training=self.is_training)
                    h1 = layers.fully_connected(self.x_ph_1, 256, scope='nn', activation_fn=tf.nn.sigmoid)
                    h2 = layers.fully_connected(self.x_ph_2, 256, reuse=True, scope='nn', activation_fn=tf.nn.sigmoid)
                    h1 = layers.batch_norm(h1, updates_collections=None, is_training=self.is_training, decay=0.5)
                    h2 = layers.batch_norm(h2, updates_collections=None, is_training=self.is_training, decay=0.5)
                    # h1 = layers.dropout(h1, keep_prob=0.2, is_training=self.is_training)
                    # h2 = layers.dropout(h2, keep_prob=0.2, is_training=self.is_training)
                    # h1 = layers.fully_connected(h1, 100, scope='nn2')
                    # h2 = layers.fully_connected(h2, 100, reuse=True, scope='nn2')
                if keep_prob < 1:
                    h11 = layers.dropout(inputs=h1, 
                                         keep_prob=0.2, 
                                         is_training=self.is_training)
                else:
                    h11 = h1

                self.h1 = h1
                self.h2 = h2
                self.pred = layers.fully_connected(h11, num_classes, activation_fn=None)
            else:
                self.x_ph_1 = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
                self.x_ph_2 = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

                @reuse('feature')
                def feature_extractor(x_ph):
                    h = layers.dropout(inputs=x_ph, keep_prob=0.9, is_training=self.is_training)
                    h = layers.conv2d(x_ph, 96, 5)
                    h = layers.max_pool2d(h, 3)
                    h = layers.dropout(inputs=h, keep_prob=0.75, is_training=self.is_training)
                    h = layers.conv2d(h, 128, 5)
                    h = layers.max_pool2d(h, 3)
                    h = layers.dropout(inputs=h, keep_prob=0.75, is_training=self.is_training)
                    h = layers.conv2d(h, 256, 5)
                    h = layers.dropout(inputs=h, keep_prob=0.5, is_training=self.is_training)
                    h = layers.flatten(h)
                    h = layers.fully_connected(h, 2048)
                    return h
                
                def classifier(h):
                    h = layers.dropout(inputs=h, keep_prob=0.5, is_training=self.is_training)
                    h = layers.fully_connected(h, 2048)
                    h = layers.dropout(inputs=h, keep_prob=0.5, is_training=self.is_training)
                    h = layers.fully_connected(h, 10, activation_fn=None)
                    return h
                
                self.h1 = feature_extractor(self.x_ph_1)
                self.h2 = feature_extractor(self.x_ph_2)
                h1 = self.h1
                h2 = self.h2
                self.pred = classifier(self.h1)

            # Losses
            if self.dist == 'l1':
                self.reg_loss = reg * distances.l1(h1, h2)
            elif self.dist == 'l1n':
                mean_h1l1 = tf.reduce_mean(tf.reduce_sum(tf.abs(h1), -1))
                mean_h2l1 = tf.reduce_mean(tf.reduce_sum(tf.abs(h2), -1))
                self.reg_loss = reg * distances.l1(h1, h2) / (mean_h1l1 + mean_h2l1)
            elif self.dist == 'l2':
                self.reg_loss = reg * distances.l2(h1, h2)
            elif self.dist == 'cos':
                self.reg_loss = -reg * distances.cos(h1, h2)
            elif self.dist == 'mmd':
                self.reg_loss = reg * distances.mmd(h1, h2, self.bandwidth)
            else:
                self.reg_loss = reg * distances.cmd(h1, h2, 5)

            clf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y_ph, logits=self.pred)
            self.clf_loss = tf.reduce_mean(clf_loss)
            self.loss = self.reg_loss + self.clf_loss
            correct = tf.nn.in_top_k(predictions=self.pred, targets=self.y_ph, k=1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            # Ops
            if self.learning_rate_0 == 0:
                print('Using AdaDelta')
                self.optimizer = tf.train.AdadeltaOptimizer(1.0)
            else:
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate_ph, 0.0)
                #self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
            self.train_op = self.optimizer.minimize(self.loss)
            self.init_op = tf.global_variables_initializer()

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.sess.run(self.init_op)

            # params = tf.trainable_variables()
            # for i in params:
            #     print(i.name, i.get_shape())


    def __enter__(self):
        print('Entering')
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        print('Exiting')
        self.sess.close()


    def fit(self, batches,
            test_tar_x=None, test_tar_y=None, test_tar_x2=None, test_tar_y2=None,
            verbose=True):

        if test_tar_x is not None:
            test_tar_x = np.copy(test_tar_x)
            test_tar_y = np.copy(test_tar_y)

        bs = self.batch_size

        loss_ma = 0
        for epoch in range(1, self.epoches + 1):
            time_epoch = -time.time()

            train_src_x, train_src_y, train_tar_x = batches.next_batch()
            train_src_y = np.argmax(train_src_y, 1)
            perm = np.random.permutation(train_src_x.shape[0])
            train_src_x = train_src_x[perm]
            train_src_y = train_src_y[perm]
            if bs == -1:
                bs = train_src_x.shape[0]

            # Perform matching
            time_match = -time.time()
            ws = []
            if not (self.dist == 'mmd' or self.dist == 'cmd' or self.reg == 0):
                d_h1, d_h2 = self.sess.run([self.h1, self.h2], 
                                      feed_dict={self.x_ph_1: train_src_x,
                                                 self.x_ph_2: train_tar_x, 
                                                 self.is_training: False})
                if self.dist == 'l1' or self.dist == 'l1n':
                    obj_matrix = distances.pw_l1(d_h1, d_h2)
                elif self.dist == 'l2':
                    obj_matrix = distances.pw_l2(d_h1, d_h2)
                elif self.dist == 'cos':
                    obj_matrix = reverse(distances.pw_cos(d_h1, d_h2))

                if self.match == 'e':
                    rs, cs = lsa(obj_matrix)
                else:
                    rs, cs = approx_lsa(obj_matrix)
                ws.append(obj_matrix[rs, cs].mean())
                train_tar_x = train_tar_x[assignments(rs, cs)]
            time_match += time.time()

            # SGD
            for t in range(train_src_x.shape[0] // bs):
                x_batch = train_src_x[t*bs : (t+1)*bs]
                x2_batch = train_tar_x[t*bs : (t+1)*bs]
                y_batch = train_src_y[t*bs : (t+1)*bs]

                lr = self.learning_rate_0 * self.t0 / (self.t0 + epoch)

                _, l, c, r, a = \
                        self.sess.run([self.train_op, 
                                       self.loss, self.clf_loss, 
                                       self.reg_loss, self.accuracy], 
                        feed_dict={self.x_ph_1: x_batch, 
                                   self.x_ph_2: x2_batch, 
                                   self.y_ph: y_batch,
                                   self.learning_rate_ph: lr,
                                   self.is_training: True})
                loss_ma = loss_ma * 0.99 + l * 0.01
                # print(c, r)

            time_epoch += time.time()

            if epoch % 20 == 0:
                if verbose:
                    print('Iteration {} ({:.2f}s, match {:.2f}s): loss = {:.4f}, {:.4f} (clf = {:.4f}, reg = {:.4f}), '
                          'W = {:.4f}, training accuracy = {:.4f}'
                            .format(epoch, time_epoch, time_match, l, loss_ma, c, r, np.mean(ws), a))

                if test_tar_x is not None:
                    _, a_test = self.predict(test_tar_x, test_tar_y)
                    if verbose:
                        print('Testing accuracy = {:.4f}'.format(a_test))
                if test_tar_x2 is not None:
                    _, a_test = self.predict(test_tar_x2, test_tar_y2)
                    if verbose:
                        print('Testing accuracy = {:.4f}'.format(a_test))


        return a_test

    def predict(self, test_tar_x, test_tar_y=None, batch_size=1000):
        if test_tar_y is not None:
            preds = []
            mean_acc = 0.0
            for t in range(test_tar_x.shape[0] // batch_size + 1):
                x_batch = test_tar_x[t*batch_size : (t+1)*batch_size]
                y_batch = test_tar_y[t*batch_size : (t+1)*batch_size]
                if x_batch.shape[0] == 0:
                    break

                pred, a = self.sess.run([self.pred, self.accuracy],
                                    feed_dict={self.x_ph_1: x_batch,
                                               self.y_ph: y_batch,
                                               self.is_training: False})
                preds.append(pred)
                mean_acc += x_batch.shape[0] * a

            return np.concatenate(preds), mean_acc / test_tar_x.shape[0]
        else:
            pred = self.sess.run(self.pred,
                                feed_dict={self.x_ph_1: test_tar_x,
                                           self.is_training: False})
            return pred

    def representation(self, x):
        return self.sess.run(self.h1, 
                feed_dict={self.x_ph_1: x})
