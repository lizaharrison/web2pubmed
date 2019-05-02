"""
Constructs shared vocabulary containing all distinct words common to both the PubMed and web corpora.
Two different vocabularies are generated:
    - FULL VOCAB (NO THRESHOLD PARAMETERS): No minimum or maximum document frequency limits applied
    during construction of vocabulary. To be used for analyses using truncated singular-value decomposition (T-SVD) to
    reduce dimensionality
    - THRESHOLD PARAMETERS: Only words appearing in at least 2 documents but less than 90% of documents are
    included in the final shared vocabulary. Alternative method of dimensionality reduction to T-SVD.
"""

import pandas as pd
import database_functions as db_fnc
import pickle
import scipy
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def train_test_corpus():

    # DATABASE DOWNLOAD
    # Established database connection
    cnxn = db_fnc.database_connect(db_fnc.db, db_fnc.host, db_fnc.user, db_fnc.password)
    all_data = db_fnc.db_to_df(db_fnc.final_corpora_scripts, cnxn)

    # Extracts each dataset (database table) from list
    pm_corpus_1 = all_data[0].set_index('pmid').sort_values('web_id')
    web_corpus_1 = all_data[1].set_index('web_id').sort_index()
    corpora_links_1 = all_data[2].set_index('web_id').sort_values('web_id')

    # Select web-PubMed links in test datasetsv (training set only used for fitting CCA model)
    # NOTE: PubMed test set contains both linked and unlinked articles [n = len(pm_corpus_1) - len(pm_corpus_train)]
    corpora_links_train = corpora_links_1.loc[corpora_links_1['train_test'] == 'Train']
    web_corpus_train = web_corpus_1.loc[web_corpus_1.index.isin(corpora_links_train.index)]
    pm_corpus_train = pm_corpus_1.loc[pm_corpus_1.index.isin(corpora_links_train['pmid'])]

    all_train_datasets = {'corpora_links_train': corpora_links_train,
                          'web_corpus_train': web_corpus_train,
                          'pm_corpus_train': pm_corpus_train,
                          }
    pickle.dump(all_train_datasets,
                open('all_train_datasets.pkl', 'wb'),
                protocol=2,
                )

    corpora_links_test = corpora_links_1.loc[corpora_links_1['train_test'] == 'Test']
    web_corpus_test = web_corpus_1.loc[~web_corpus_1.index.isin(corpora_links_train.index)]
    pm_corpus_test = pm_corpus_1.loc[~pm_corpus_1.index.isin(corpora_links_train['pmid'])]
    all_test_datasets = {'corpora_links_test': corpora_links_test,
                         'web_corpus_test': web_corpus_test,
                         'pm_corpus_test': pm_corpus_test,
                         }
    pickle.dump(all_test_datasets,
                open('all_test_datasets.pkl', 'wb'),
                protocol=2,
                )

    # Generates full combined corpus appending all sample web articles to all PubMed articles
    # Specific order to allow for later separation into individual web/PubMed training and test datasets
    # full_corpus_train = web_corpus_train['corpus_text'].append(pm_corpus_train['corpus_text'])
    # full_corpus_test = web_corpus_test['corpus_text'].append(pm_corpus_test['corpus_text'])

    full_web = web_corpus_train['corpus_text'].append(web_corpus_test['corpus_text'])
    full_pm = pm_corpus_train['corpus_text'].append(pm_corpus_test['corpus_text'])
    pickle.dump({'full_web': full_web,
                 'full_pm': full_pm,
                 },
                open('full_web_pm_matrix_order.pkl', 'wb'),
                protocol=2,
                )

    full_corpus = full_web.append(full_pm)

    corpus_lengths = {'web_corpus_train': len(web_corpus_train),
                      'web_corpus_test': len(web_corpus_test),
                      'pm_corpus_train': len(pm_corpus_train),
                      'pm_corpus_test': len(pm_corpus_test),
                      'full_web': len(full_web),
                      'full_pm': len(full_pm,)
                      }
    pickle.dump(corpus_lengths,
                open('test_train_corpus_lengths.pkl', 'wb'),
                protocol=2,
                )

    pd.to_pickle(full_corpus,
                 'full_web_pm_corpus_text.pkl',
                 protocol=2,
                 )

    print('%s web documents + %s PubMed documents in training set' % (len(web_corpus_train),
                                                                      len(pm_corpus_train)))
    print('%s web documents + %s PubMed documents in test set' % (len(web_corpus_test),
                                                                  len(pm_corpus_test)))
    print('There are %s known links in the training set and %s known links in the test set' % (len(corpora_links_train),
                                                                                               len(corpora_links_test)))
    return_vals = (pm_corpus_1,
                   web_corpus_1,
                   full_corpus,
                   corpora_links_1,
                   )

    return return_vals


def vocab_gen(pm_corpus,
              web_corpus,
              full_corpus,
              params_dict,
              ):

    dimensionality_reduction = params_dict['Dimensionality Reduction']
    min_df = params_dict['Min DF']
    max_df = params_dict['Max DF']

    # Generates web article vocabulary count matrix & vocabulary list
    vectorizer = CountVectorizer(lowercase=True,
                                 )
    web_countx = vectorizer.fit_transform(web_corpus['corpus_text'])
    web_vocab = pd.Series(vectorizer.get_feature_names())
    print('Web vocabulary size: ' + str(len(web_vocab)))

    # Generates PubMed article vocabulary count matrix & vocabulary list
    vectorizer = CountVectorizer(lowercase=True,
                                 )
    pubmed_countx = vectorizer.fit_transform(pm_corpus['corpus_text'])
    pubmed_vocab = pd.Series(vectorizer.get_feature_names())
    print('PubMed vocabulary size: ' + str(len(pubmed_vocab)))

    if dimensionality_reduction == 'Thresholds':
        # Generates full, combined corpus vocabulary with NO THRESHOLD PARAMETERS
        vectorizer = CountVectorizer(lowercase=True,
                                     min_df=min_df,  # Must appear in >= x articles (web or PubMed)
                                     max_df=max_df,  # Must appear in <= yy% of articles (web or PubMed)
                                     )
        web_pm_countx = vectorizer.fit_transform(full_corpus)
        web_pm_vocab = pd.Series(vectorizer.get_feature_names())
        print('Full combined corpus vocabulary size (threshold values): ' + str(len(web_pm_vocab)))

        filename = 'shared_vocab_no_thresholds.pkl'

    else:
        # Generates full, combined corpus vocabulary with NO THRESHOLD PARAMETERS
        # Used in experimental groups with dimensionality reduction using truncated singular value decomposition
        vectorizer = CountVectorizer(lowercase=True,
                                     )
        web_pm_countx = vectorizer.fit_transform(full_corpus)
        web_pm_vocab = pd.Series(vectorizer.get_feature_names())
        print('Full combined corpus vocabulary size (no threshold values): ' + str(len(web_pm_vocab)))

        filename = 'shared_vocab_thresholds_%s_%s.pkl' % (min_df, min_df)

    # Identifies words appearing in both corpora, and those appearing in only one corpus (web or PubMed)
    shared_vocab_no_threshold = web_pm_vocab[web_pm_vocab.isin(web_vocab) & web_pm_vocab.isin(pubmed_vocab)]
    # web_only_vocab_no_threshold = web_vocab[~web_vocab.isin(pubmed_vocab)]
    # pubmed_only_vocab_no_threshold = pubmed_vocab[~pubmed_vocab.isin(web_vocab)]

    # Identifies and removes all 'numeric-only' words in vocabulary
    numeric_words = [word for word in set(shared_vocab_no_threshold) if word.isdigit()]
    shared_vocab = shared_vocab_no_threshold[~shared_vocab_no_threshold.isin(numeric_words)].reset_index(drop=True,
                                                                                                                                   )
    print('Final vocabulary size: ' + str(len(shared_vocab_no_threshold)))

    # Saves NO THRESHOLD vocabulary to file for use in analyses
    pd.to_pickle(shared_vocab,
                 filename,
                 protocol=2,
                 )

    return_vals = shared_vocab

    return return_vals


def feature_representation(vocab,
                           full_corpus,
                           params_dict,
                           ):

    feature_extraction = params_dict['Feature Extraction']
    dimensionality_reduction = params_dict['Dimensionality Reduction']
    min_df = params_dict['Min DF']
    max_df = params_dict['Max DF']
    tsvd_components = params_dict['T-SVD Components']

    if dimensionality_reduction == 'Thresholds':
        matrix_filename = '%s_matrix_%s_%s_%s.npz' % (feature_extraction,
                                                      dimensionality_reduction,
                                                      min_df,
                                                      max_df,
                                                      )
        vectorizer_filename = '%s_vectorizer_%s_%s_%s.pkl' % (feature_extraction,
                                                              dimensionality_reduction,
                                                              min_df,
                                                              max_df,
                                                              )

    else:
        matrix_filename = '%s_matrix_%s.npz' % (feature_extraction,
                                                dimensionality_reduction,
                                                )
        vectorizer_filename = '%s_vectorizer_%s.pkl' % (feature_extraction,
                                                        dimensionality_reduction,
                                                        )

    if feature_extraction == 'Binary':
        # Vectorization of full, combined corpus (PubMed + web)
        print('Beginning binary vectorization (NO THRESHOLDS)...')
        vectorizer = CountVectorizer(vocabulary=vocab.values,
                                     binary=True,
                                     )
        print(vectorizer)
        binary_x = vectorizer.fit_transform(full_corpus)
        print('Binary vectorization complete '
              + str(binary_x.shape))
        # Saves sparse matrix and vectorizer objects to files
        scipy.sparse.save_npz(matrix_filename, binary_x)
        pickle.dump(vectorizer,
                    open(vectorizer_filename, 'wb'),
                    protocol=2,
                    )

    elif feature_extraction == 'TF':
        # Term frequency vectorization  of full, combined corpus (PubMed + web)
        print('Beginning term frequency vectorization (NO THRESHOLDS)...')
        vectorizer = CountVectorizer(vocabulary=vocab.values,
                                     )
        print(vectorizer)
        tf_x = vectorizer.fit_transform(full_corpus)
        print('Term frequency vectorization complete '
              + str(tf_x.shape))

        # Saves sparse matrix and vectorizer objects to files
        scipy.sparse.save_npz(matrix_filename, tf_x)
        pickle.dump(vectorizer,
                    open(vectorizer_filename, 'wb'),
                    protocol=2,
                    )

    else:
        # Term frequency vectorization  of full, combined corpus (PubMed + web)
        print('Beginning term frequency vectorization (NO THRESHOLDS)...')
        vectorizer = TfidfVectorizer(vocabulary=vocab.values,
                                     )
        print(vectorizer)
        tfidf_x = vectorizer.fit_transform(full_corpus)
        print('Term frequency vectorization complete '
              + str(tfidf_x.shape))

        # Saves sparse matrix and vectorizer objects to files
        scipy.sparse.save_npz(matrix_filename, tfidf_x)
        pickle.dump(vectorizer,
                    open(vectorizer_filename, 'wb'),
                    protocol=2,
                    )
        print('TF-IDF vectorization complete '
              + str(tfidf_x.shape))

    print('Feature matrix saved to file')
