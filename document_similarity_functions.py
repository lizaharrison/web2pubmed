import os
import pandas as pd
import pickle
import scipy
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# Defines function for splitting matrices into component parts
def matrix_split(feature_matrix, part_1_n):
    """
    Splits a single sparse matrix along the row axis into two parts of
    size n and matrix_len - n.

    Args:
        feature_matrix: Sparse matrix
            Full matrix to be broken into parts of n-size
        part_1_n: int
            Size of first part (row at which matrix is split)

    Returns:
        return_vals: Tuple containing newly generated parts of original matrix

    """

    # Selects first part_1_n rows in the feature matrix
    part_1_x = scipy.sparse.csr_matrix(feature_matrix[:part_1_n])

    # Selects the remaining rows in feature matrix
    part_2_x = scipy.sparse.csr_matrix(feature_matrix[part_1_n:])

    return_vals = (part_1_x, part_2_x)

    return return_vals


def tsvd_function(sparse_matrix, n_components):
    """
    Performs Truncated singular value decomposition of a sparse matrix, reducing
    dimensionality to a specified number of components
    Args:
        sparse_matrix: Sparse matrix
        Sparse matrix on which to perform T-SVD
        n_components: int
            Number of singular values to retain in final TSVD matrix

    Returns:
        return_vals: Tuple containing TSVD function object and TSVD matrix

    """
    tsvd_obj = TruncatedSVD(n_components=n_components)
    tsvd_x = tsvd_obj.fit_transform(sparse_matrix)
    return_vals = (tsvd_obj,
                   scipy.sparse.csr_matrix(tsvd_x),
                   )

    print('Sum of explained variance ratio = ' + str(tsvd_obj.explained_variance_ratio_.sum()))
    print(tsvd_obj.explained_variance_ratio_)

    return return_vals


#####


def experimental_groups_gen(feature_extraction_methods,
                            min_dfs,
                            max_dfs,
                            tsvd_components,
                            cca_components,
                            ):
    feature_reps_counts = len(feature_extraction_methods)
    threshold_counts = len(min_dfs)
    tsvd_counts = len(tsvd_components)
    cca_counts = len(cca_components)

    if cca_counts == 0:
        all_groups_count = (threshold_counts + tsvd_counts) * feature_reps_counts

        feature_extraction_all = [feature_xt for feature_xt in feature_extraction_methods] * (threshold_counts +
                                                                                              tsvd_counts
                                                                                              )
        min_df_all = [min_df for min_df in min_dfs] * feature_reps_counts
        max_df_all = [max_df for max_df in max_dfs] * feature_reps_counts
        min_df_all.extend([2] * (tsvd_counts * feature_reps_counts))
        max_df_all.extend([1] * (tsvd_counts * feature_reps_counts))

        dimensionality_reduction_all = ['Thresholds'] * (threshold_counts * feature_reps_counts)
        dimensionality_reduction_all.extend(['T-SVD'] * ((tsvd_counts + cca_counts) * feature_reps_counts))
        dimensionality_reduction_all.sort(reverse=True)

        tsvd_components_all = [0] * (threshold_counts * feature_reps_counts) + \
                              [components for components in tsvd_components] * feature_reps_counts
        tsvd_components_all.sort()

        cca_components_all = [0] * all_groups_count
        cca_all = [False] * all_groups_count

    else:
        cca_components_all = []
        dimensionality_reduction_all = []
        tsvd_components_all = []

        for tsvd in tsvd_components:
            print(tsvd)
            tsvd_cca = [cca for cca in cca_components if cca <= tsvd]
            tsvd_cca.sort()
            print(tsvd_cca)
            cca_components_all.extend(tsvd_cca)
            dimensionality_reduction_all.extend(['T-SVD'] * len(tsvd_cca))
            tsvd_components_all.extend([tsvd] * len(tsvd_cca))

        feature_extraction_all = feature_extraction_methods * len(cca_components_all)

        cca_all = [True] * len(cca_components_all)

        min_df_all = [min_df for min_df in min_dfs] * feature_reps_counts
        max_df_all = [max_df for max_df in max_dfs] * feature_reps_counts
        min_df_all.extend([2] * len(cca_components_all))
        max_df_all.extend([1] * len(cca_components_all))

    all_param_combos = list(zip(feature_extraction_all,
                                dimensionality_reduction_all,
                                min_df_all,
                                max_df_all,
                                tsvd_components_all,
                                cca_all,
                                cca_components_all,
                                ))

    all_experimental_groups = []
    exp_groups_params_list = ['Feature Extraction',
                              'Dimensionality Reduction',
                              'Min DF',
                              'Max DF',
                              'T-SVD Components',
                              'CCA',
                              'CCA Components',
                              ]

    for params in all_param_combos:
        all_experimental_groups.append(dict(zip(exp_groups_params_list, params)))

    [print(x) for x in all_experimental_groups]

    return all_experimental_groups


def train_test_matrix(corpus_lengths,
                      params_dict,
                      ):
    print(os.getcwd())

    feature_extraction = params_dict['Feature Extraction']
    dimensionality_reduction = params_dict['Dimensionality Reduction']
    apply_cca = params_dict['CCA']
    tsvd_components = params_dict['T-SVD Components']
    cca_components = params_dict['CCA Components']
    min_df = params_dict['Min DF']
    max_df = params_dict['Max DF']

    if dimensionality_reduction == 'Thresholds':
        matrix_filename = '%s_matrix_%s_%s_%s.npz' % (feature_extraction,
                                                      dimensionality_reduction,
                                                      min_df,
                                                      max_df,
                                                      )

        # Loads full feature matrices from file depending on feature extraction and dimensionality reduction methods
        feature_x_full = scipy.sparse.load_npz(matrix_filename)
        feature_matrix = feature_x_full

    else:
        matrix_filename = '%s_matrix_%s.npz' % (feature_extraction,
                                                dimensionality_reduction,
                                                )
        # Loads full feature matrices from file depending on feature extraction and dimensionality reduction methods
        feature_x_full = scipy.sparse.load_npz(matrix_filename)
        print('%s feature matrix loaded from file ' % feature_extraction)

        print('Beginning T-SVD...')
        tsvd_object, feature_matrix = tsvd_function(feature_x_full, tsvd_components)
        scipy.sparse.save_npz('%s_T-SVD_Applied_%s_matrix.npz' % (feature_extraction,
                                                                  tsvd_components,
                                                                  ),
                              feature_matrix,
                              )
        print('T-SVD dimensionality reduction complete')

    #####

    # TRAIN & TEST DATASETS #
    print('Beginning generation of TRAIN and TEST datasets...')

    # Splits feature matrix into parts cointaining web and PubMed vectors
    web_feature_matrix, pm_feature_matrix = matrix_split(feature_matrix, corpus_lengths['full_web'])

    # Splits web matrix in to train and test datasets
    web_train_matrix, web_test_matrix = matrix_split(web_feature_matrix, corpus_lengths['web_corpus_train'])

    scipy.sparse.save_npz(matrix_filename + '_web_train.npz',
                          web_train_matrix,
                          )
    scipy.sparse.save_npz(matrix_filename + '_web_test.npz',
                          web_test_matrix,
                          )

    # Splits PubMed matrix in to train and test datasets
    pm_train_matrix, pm_test_matrix = matrix_split(pm_feature_matrix, corpus_lengths['pm_corpus_train'])

    scipy.sparse.save_npz(matrix_filename + '_pm_train.npz',
                          pm_train_matrix,
                          )
    scipy.sparse.save_npz(matrix_filename + '_pm_test.npz',
                          pm_test_matrix,
                          )

    print('TRAIN and TEST datasets generated')
    all_test_train_data = (web_train_matrix,
                           web_test_matrix,
                           pm_train_matrix,
                           pm_test_matrix,
                           )

    return all_test_train_data


def cosine_sim(web_test_matrix,
               pm_test_matrix,
               web_test_corpus,
               pm_test_corpus,
               known_links_corpus,
               params_dict,
               ):
    feature_extraction = params_dict['Feature Extraction']
    dimensionality_reduction = params_dict['Dimensionality Reduction']
    apply_cca = params_dict['CCA']
    tsvd_components = params_dict['T-SVD Components']
    cca_components = params_dict['CCA Components']

    #####

    # COSINE SIMILARITY #
    print('Beginning cosine similarity calculations...')

    # Cosine distance between web and PubMed vectors in tf-idf representation following transformation using CCA
    cosine_matrix = cosine_similarity(web_test_matrix, pm_test_matrix)
    print('Cosine similarity complete:\n'
          + str(cosine_matrix.shape))

    # Convert to dataframe
    cosine_df = pd.DataFrame(cosine_matrix,
                             index=web_test_corpus.index,
                             columns=pm_test_corpus.index,
                             )

    #####

    # RANKING OF CORRECT LINKS #
    print('Beginning ranking...')
    correct_ranks_all = []

    for index in known_links_corpus.index:
        # Saves web_id and PMID for each link to objects
        web_id = index
        pmid = known_links_corpus.loc[index, 'pmid']

        # Ranks all cosine distances between vectors in BINARY representation
        ranks = cosine_df.loc[web_id].sort_values().rank(axis=0,
                                                         method='min',
                                                         ascending=False,
                                                         )
        correct_link_rank = ranks[pmid]
        correct_ranks_all.append(correct_link_rank)

    cos_col_name = '%s_%s_%s' % (feature_extraction,
                                 dimensionality_reduction,
                                 tsvd_components,
                                 )
    correct_links_srs = pd.Series(correct_ranks_all,
                                  index=known_links_corpus.index,
                                  name=cos_col_name,
                                  )

    '''
    if apply_cca is False:
        pickle.dump(known_links_corpus,
                    open('%s_%s_%s_%s-tsvd_ranks.pkl' % (feature_extraction,
                                                         dimensionality_reduction,
                                                         tsvd_components,
                                                         cca_components,
                                                         ), 'wb'),
                    protocol=2,
                    )
    else:
        pickle.dump(known_links_corpus,
                    open('%s_%s_%s_%s-tsvd_%s-cca_ranks.pkl' % (feature_extraction,
                                                                dimensionality_reduction,
                                                                apply_cca,
                                                                tsvd_components,
                                                                cca_components,
                                                                ), 'wb'),
                    protocol=2,
                    )

        print('Final results saved to file')
    '''

    print('All cosine similarity scores ranked')

    return correct_links_srs


def measures(correct_links_srs,
             params_dict,
             ):
    feature_extraction = params_dict['Feature Extraction']
    dimensionality_reduction = params_dict['Dimensionality Reduction']
    apply_cca = params_dict['CCA']
    tsvd_components = params_dict['T-SVD Components']
    cca_components = params_dict['CCA Components']

    # METRIC 1) MEDIAN RANK
    cosine_metrics = correct_links_srs.describe()

    median = cosine_metrics['50%']
    iqr_25 = cosine_metrics['25%']
    iqr_75 = cosine_metrics['75%']

    # METRIC 2) PERCENTAGE CORRECT
    cosine_correct = round(len(correct_links_srs.loc[correct_links_srs == 1]) /
                           len(correct_links_srs) * 100, 2)

    # METRIC 3) PERCENTAGE IN TOP 50
    cosine_top_50 = round((len(correct_links_srs.loc[correct_links_srs <= 50]) /
                           len(correct_links_srs)) * 100, 2)

    # METRIC 3) PERCENTAGE IN TOP 100
    cosine_top_250 = round((len(correct_links_srs.loc[correct_links_srs <= 250]) /
                            len(correct_links_srs)) * 100, 2)

    print('Median rank (IQR): %s (%s - %s)' % (median, iqr_25, iqr_75))
    print('Percentage of links ranked correctly: %s' % cosine_correct)
    print('Percentage of links ranked in top 50: %s' % cosine_top_50)
    print('Percentage of links ranked in top 250: %s' % cosine_top_250)

    results_data = {'Feature Extraction': feature_extraction,
                    'Dimensionality Reduction': dimensionality_reduction,
                    'Min DF': params_dict['Min DF'],
                    'Max DF': params_dict['Max DF'],
                    'T-SVD Components': tsvd_components,
                    'CCA': apply_cca,
                    'CCA Components': cca_components,
                    'Median Rank': median,
                    'IQR_25': iqr_25,
                    'IQR_75': iqr_75,
                    'Percentage correct': cosine_correct,
                    'Percentage in Top 50': cosine_top_50,
                    'Percentage in Top 250': cosine_top_250,
                    }
    if dimensionality_reduction == 'Thresholds':
        pickle.dump(results_data,
                    open('%s_%s_%s_%s_results.pkl' % (feature_extraction,
                                                      dimensionality_reduction,
                                                      params_dict['Min DF'],
                                                      params_dict['Max DF'],
                                                      ), 'wb'),
                    protocol=2,
                    )
    else:
        if apply_cca is False:
            pickle.dump(results_data,
                        open('%s_%s_%s_results.pkl' % (feature_extraction,
                                                       dimensionality_reduction,
                                                       tsvd_components,
                                                       ), 'wb'),
                        protocol=2,
                        )
        else:
            pickle.dump(results_data,
                        open('%s_%s_%s_%s-cca_results.pkl' % (feature_extraction,
                                                              dimensionality_reduction,
                                                              tsvd_components,
                                                              cca_components,
                                                              ), 'wb'),
                        protocol=2,
                        )
    print('Final results saved to file')

    return results_data


'''
    #####

    # CANONICAL COVARIATE ANALYSIS #

    if apply_cca is True:
        print('Beginning CCA...')

        web_train_matrix = web_train_matrix.todense()
        web_test_matrix = web_test_matrix.todense()

        pm_train_matrix = pm_train_matrix.todense()
        pm_test_matrix = pm_test_matrix.todense()

        # Fits CCA model to TRAIN data
        apply_cca = CCA(n_components=cca_components)
        apply_cca.fit(web_train_matrix,
                      pm_train_matrix,
                      )
        print('CCA model fitted to TRAINING data')

        """
        # Transforms TRAIN data using fitted model
        web_train_matrix_2, pm_train_matrix_2 = apply_cca.transform(web_train_matrix,
                                                                    pm_train_matrix,
                                                                    )
        print('Transformation of TRAINING data using fitted CCA model complete')
        """

        # Transforms TEST data using fitted model
        web_test_matrix_2, pm_test_matrix_2 = apply_cca.transform(web_test_matrix,
                                                                  pm_test_matrix,
                                                                  )
        print('Transformation of TESTING data using fitted CCA model complete')

    elif apply_cca is False:
        # NO CCA REQUIRED
        web_test_matrix_2 = web_test_matrix
        pm_test_matrix_2 = pm_test_matrix

    else:
        raise ValueError('CCA must be set to True or False')
'''