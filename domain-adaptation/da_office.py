import sys
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from six.moves import range
import numpy as np

from scipy.io import loadmat
import sklearn
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from pmd import PMD_DA as PMD
import seaborn as sns
from batch_generator import BatchGenerator


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('src', 0, 'Source domain')
tf.app.flags.DEFINE_integer('tar', 2, 'Target domain')
tf.app.flags.DEFINE_integer('epoches', 2000, 'Number of training epoches')
tf.app.flags.DEFINE_float('reg', 0.0, 'Number of training epoches')
tf.app.flags.DEFINE_float('keep_prob', 0.2, 'Probability of keeping')
tf.app.flags.DEFINE_string('dist', 'l1', 'Type of distance: l1, l2, cos or mmd')
tf.app.flags.DEFINE_float('bandwidth', 1.0, 'MMD bandwidth')
tf.app.flags.DEFINE_float('lr0', 0, 'Learning rate, 0 for AdaDelta')
tf.app.flags.DEFINE_integer('bs', -1, 'Batch size')
tf.app.flags.DEFINE_integer('mbs', 500, 'Match batch size')
tf.app.flags.DEFINE_string('embedding', 'none', 'Name of the output embedding file.')
tf.app.flags.DEFINE_bool('batch_norm', False, 'Batch normalization')
tf.app.flags.DEFINE_string('exp', 'pmd', 'Experiment')

def run_svm(train_x, train_y, test_x, test_y):
    # Cross validate the best C
    cs = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
    accs = np.zeros(cs.shape)
    for i in range(cs.shape[0]):
        t = -time.time()
        clf = LinearSVC(C=cs[i])
        accs[i] = cross_val_score(clf, train_x, train_y).mean()
        t += time.time()
        print('Running C={} ({:.2f} seconds): {}'.format(cs[i], t, accs[i]))

    bestC = cs[np.argmax(accs)]

    clf = LinearSVC(C = bestC)
    model = clf.fit(train_x, train_y)
    train_accuracy = clf.score(train_x, train_y)
    test_accuracy = clf.score(test_x, test_y)
    print('BestC = {}, SVM training accuracy = {}, testing accuracy = {}'
            .format(bestC, train_accuracy, test_accuracy))


def run_experiment(src, tar, validate):
    reg = FLAGS.reg
    keep_prob = FLAGS.keep_prob
    domains = ['amazon', 'dslr', 'webcam']

    src_x = np.load('data/{}_img_repr.npy'.format(domains[src]))
    src_y = np.load('data/{}_labels.npy'.format(domains[src]))
    tar_x = np.load('data/{}_img_repr.npy'.format(domains[tar]))
    tar_y = np.load('data/{}_labels.npy'.format(domains[tar]))

    valid_x = tar_x[:100]
    valid_y = np.argmax(tar_y[:100], 1)

    # batches = Batches(src_x, src_y, tar_x, tar_x.shape[0])
    batches = BatchGenerator(src_x, src_y, tar_x, FLAGS.mbs)
    src_y = np.argmax(src_y, 1)
    tar_y = np.argmax(tar_y, 1)

    # Parameters
    n_x = src_x.shape[1]
    n_h = 256
    num_classes = 31
    learning_rate_0 = FLAGS.lr0
    t0 = 100
    epoches = FLAGS.epoches

    print('{} -> {}, reg = {}, bw = {}, keep_prob = {}, lr = {}, dist = {}'.format(domains[src], domains[tar], reg, FLAGS.bandwidth, keep_prob, learning_rate_0, FLAGS.dist))
    # run_svm(src_x, src_y, tar_x, tar_y)
    # sys.exit(0)

    params = {'n_x': n_x, 'n_h': n_h, 'num_classes': num_classes,
              'reg': reg, 'keep_prob': keep_prob,
              'learning_rate_0': learning_rate_0, 't0': t0,
              'epoches': epoches, 
              'batch_size': FLAGS.bs,
              'dist': FLAGS.dist, 'bn': FLAGS.batch_norm}
    if FLAGS.dist == 'mmd':
        params['bandwidth'] = FLAGS.bandwidth

    with PMD(params) as pmd:
        a_test = pmd.fit(batches, tar_x, tar_y)
        if validate:
            _, a_valid = pmd.predict(valid_x, valid_y)
            print('T accuracy = {}, V accuracy = {}'.format(a_test, a_valid))
    #     if validate:
    #         valid_batches, valid_test_x, valid_test_y = \
    #                 batches.generate_validate(pmd)
    # if validate:
    #     with PMD(params) as valid_model:
    #         a_valid = valid_model.fit(valid_batches, valid_test_x, valid_test_y)
    #     print('T accuracy = {}, V accuracy = {}'.format(a_test, a_valid))

    if FLAGS.embedding != 'none':
        print('Fitting Embedding...')
        embed_time = -time.time()
        src_rep = pmd.representation(src_x)
        tar_rep = pmd.representation(tar_x)
        np.random.shuffle(src_rep)
        np.random.shuffle(tar_rep)
        reps = np.vstack((src_rep, tar_rep))
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        reps = tsne.fit_transform(reps)
        src_emb = reps[:src_rep.shape[0]]
        tar_emb = reps[src_rep.shape[0]:]
        print(src_rep.shape, tar_rep.shape, src_emb.shape, tar_emb.shape)
        plt.plot(src_emb[:,0], src_emb[:,1], 'b.')
        plt.plot(tar_emb[:,0], tar_emb[:,1], 'r.')
        plt.savefig(FLAGS.embedding + '.png')
        embed_time += time.time()
        print('Finished in {:.2f} seconds'.format(embed_time))

    return a_test, a_valid


def run_nn():
    FLAGS.reg = 0
    for i in range(10):
        a_test, a_valid = run_experiment(FLAGS.src, FLAGS.tar, True)
        print('T accuracy = {}, v accuracy = {}'.format(a_test, a_valid))


def run_mmd():
    FLAGS.dist = 'mmd'
    best_accuracy = 0.0
    best_reg = 0
    best_bw = 0
    for reg in [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]:
        for bw in [0.25, 1, 4, 9, 16, 25, 36, 49]:
            FLAGS.reg = reg
            FLAGS.bandwidth = bw
            a_test, a_valid = run_experiment(FLAGS.src, FLAGS.tar, True)
            print('T accuracy = {}, v accuracy = {}'.format(a_test, a_valid))
            if a_valid > best_accuracy:
                best_accuracy = a_valid
                best_reg = reg
                best_bw = bw
                print('Current best reg = {}, bw = {}, acc = {}'
                        .format(best_reg, best_bw, best_accuracy))
    FLAGS.reg = best_reg
    FLAGS.bandwidth = best_bw
    for rep in range(10):
        a_test = run_experiment(FLAGS.src, FLAGS.tar, False)
        print('Test accuracy = {}'.format(a_test))


def run_pmd():
    best_accuracy = 0.0
    best_reg = 0
    for reg in [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
    # for reg in [0.003]:
        FLAGS.reg = reg
        a_test, a_valid = run_experiment(FLAGS.src, FLAGS.tar, True)
        print('T accuracy = {}, v accuracy = {}'.format(a_test, a_valid))
        if a_valid > best_accuracy:
            best_accuracy = a_valid
            best_reg = reg
            print('Current best reg = {}, acc = {}'
                    .format(best_reg, best_accuracy))
    FLAGS.reg = best_reg
    tas = []
    for rep in range(10):
        a_test = run_experiment(FLAGS.src, FLAGS.tar, False)
        tas.append(a_test)
        print('Test accuracy = {}'.format(a_test))
    print(np.mean(tas), np.std(tas))


def main(argv=None):
    np.random.seed(1234)

    run_experiment(FLAGS.src, FLAGS.tar, True)
    # if FLAGS.exp == 'pmd':
    #     run_pmd()
    # else:
    #     run_mmd()
    # run_nn()

    # src = FLAGS.src
    # tar = FLAGS.tar
    # if src != -1:
    #     run_experiment(src, tar, False)
    # else:
    #     for src in range(4):
    #         for tar in range(4):
    #             if src != tar:
    #                 run_experiment(src, tar)


if __name__ == '__main__':
    tf.app.run()
