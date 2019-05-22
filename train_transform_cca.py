# -*- coding: utf-8 -*-
# Author: Didi Surian

from __future__ import division
from sklearn.cross_decomposition import CCA
import time
import scipy.sparse
from scipy.sparse import csr_matrix
import glob
import numpy as np
import sys
import _pickle as pickle



def write_to_file(outFile, text, mode):
    with open(outFile, mode) as oF:
        oF.write(text)


def dump_var(data, outFile):
    pickle.dump(data, open(outFile, "wb"), protocol=2)


def load_var(inFile):
    return pickle.load(open(inFile, "rb"))


def run_cca(l_cca_components, X_train, Y_train, X_test, Y_test, fname, tsvddim, dataFolder, logFile):
    for cca_components in l_cca_components:
        try:
            print('\nCCA components: {0}'.format(cca_components))

            write_to_file(logFile, 'CCA components: ' + str(cca_components) + '...', 'a')

            cca = CCA(n_components=cca_components, max_iter=5000)

            start_time = time.time()
            print('\tFit...'),
            cca.fit(X_train.toarray(), Y_train.toarray())
            partTime = (time.time() - start_time)
            print('done in {0} secs'.format(partTime))

            start_time = time.time()
            print('\tDump model...'),
            dump_var(cca, 'cca_' + str(cca_components) + '_fortsvddim_'+ str(tsvddim) + '.cpickle')
            partTime = (time.time() - start_time)
            print ('done in {0} secs'.format(partTime))


            start_time = time.time()
            print('\tTransform train...'),
            X_train_transform, Y_train_transform = cca.transform(X_train.toarray(), Y_train.toarray())
            partTime = (time.time() - start_time)
            print('done in {0} secs'.format(partTime))

            start_time = time.time()
            print('\tTransform test...'),
            X_test_transform, Y_test_transform = cca.transform(X_test.toarray(), Y_test.toarray())
            partTime = (time.time() - start_time)
            print('done in {0} secs'.format(partTime))

            start_time = time.time()
            print('\tDump transformed train...'),
            scipy.sparse.save_npz(dataFolder + fname + str(tsvddim) + '_cca_' + str(cca_components) + '_web_train.npz',
                                  scipy.sparse.csr_matrix(X_train_transform))
            scipy.sparse.save_npz(dataFolder + fname + str(tsvddim) + '_cca_' + str(cca_components) + '_pm_train.npz',
                                  scipy.sparse.csr_matrix(Y_train_transform))
            partTime = (time.time() - start_time)
            print('done in {0} secs'.format(partTime))

            start_time = time.time()
            print('\tDump transformed test...'),
            scipy.sparse.save_npz(dataFolder + fname + str(tsvddim) + '_cca_' + str(cca_components) + '_web_test.npz',
                                  scipy.sparse.csr_matrix(X_test_transform))
            scipy.sparse.save_npz(dataFolder + fname + str(tsvddim) + '_cca_' + str(cca_components) + '_pm_test.npz',
                                  scipy.sparse.csr_matrix(Y_test_transform))
            partTime = (time.time() - start_time)
            print('done in {0} secs'.format(partTime))

            write_to_file(logFile, 'done\n', 'a')

        except Exception as e:
            print('\n[ERROR] Component {0}. Error: {1}'.format(cca_components, e))

            write_to_file(logFile, 'Error: ' + str(e) + '\n', 'a')
            continue


if __name__ == '__main__':
    dataFolder = 'Dataset/'
    fname = 'TF-IDF_T-SVD_Applied_'

    logFile = 'log.txt'
    write_to_file(logFile,'','w')

    corpus_lengths = {'web_corpus_train': 2501,
                      'web_corpus_test': 1072,
                      'pm_corpus_train': 2501,
                      'pm_corpus_test': 205037,
                      'full_web': 3573,
                      'full_pm': 207538}

    # l_tsvddim_l_ccacomponents = [[400, [100, 200, 400]],
    #                              [800, [100, 200, 400, 800]],
    #                              [1600, [100, 200, 400, 800, 1600]]]

    l_tsvddim_l_ccacomponents = [[100, [50, 100]],
                                 [200, [50, 100, 200]],
                                 [400, [50]],
                                 [800, [50]],
                                 [1600, [50]]]

    for tc in l_tsvddim_l_ccacomponents:
        tsvddim = tc[0]
        l_cca_components = tc[1]

        print('\n----------------------------')
        print('TSVD dimension: {0}'.format(tsvddim))

        st_log = '\n----------------------------\n'
        st_log += 'TSVD dimension: ' + str(tsvddim) + '\n'

        #-- Loads tfidf feature matrix from file
        print('- Load file...'),
        start_time = time.time()
        inFile = dataFolder + fname + str(tsvddim) + '_matrix.npz'
        # tfidf_x = scipy.sparse.load_npz(glob.glob(inFile)[0])      # this is csr matrix
        tfidf_x = scipy.sparse.load_npz(inFile)  # this is csr matrix
        partTime = (time.time() - start_time)
        print('done in {0} secs'.format(partTime))

        print('\tShape: {0}'.format(tfidf_x.shape))

        st_log += '\ttfidf_x.shape: ' + str(tfidf_x.shape) + '\n'

        web_tfidf_x = tfidf_x[:corpus_lengths['full_web']]
        pm_tfidf_x = tfidf_x[corpus_lengths['full_web']:]

        web_tfidf_x_train = web_tfidf_x[:corpus_lengths['web_corpus_train']]
        web_tfidf_x_test = web_tfidf_x[corpus_lengths['web_corpus_train']:]
        pm_tfidf_x_train = pm_tfidf_x[:corpus_lengths['pm_corpus_train']]
        pm_tfidf_x_test = pm_tfidf_x[corpus_lengths['pm_corpus_train']:]

        tfidf_x = None
        del tfidf_x
        web_tfidf_x = None
        del web_tfidf_x
        pm_tfidf_x = None
        del pm_tfidf_x

        print('')
        print('\tWeb train: {0}'.format(web_tfidf_x_train.shape))
        st_log += '\tWeb train: ' + str(web_tfidf_x_train.shape) + '\n'
        print ('\t    test: {0}'.format(web_tfidf_x_test.shape))
        st_log += '\t    test: ' + str(web_tfidf_x_test.shape) + '\n'
        print('\tPM train: {0}'.format(pm_tfidf_x_train.shape))
        st_log += '\tPM train: ' + str(pm_tfidf_x_train.shape) + '\n'
        print('\t   test: {0}'.format(pm_tfidf_x_test.shape))
        st_log += '\t   test: ' + str(pm_tfidf_x_test.shape) + '\n'

        write_to_file(logFile, st_log, 'a')




        run_cca(l_cca_components,
                web_tfidf_x_train, pm_tfidf_x_train, web_tfidf_x_test, pm_tfidf_x_test,
                fname, tsvddim, dataFolder, logFile)

        # tsvddim = 400
        # cca_components = 100
        # a = scipy.sparse.load_npz(dataFolder + fname + str(tsvddim) + '_cca_' + str(cca_components) + 'pm_train.npz')

        print('')

