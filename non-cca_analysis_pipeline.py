import os
import config
import time
import pickle
import pandas as pd
import feature_extraction_functions as ft_x
import document_similarity_functions as doc_sim
from collections import OrderedDict

#####

# SET-UP
wd = config.working_dir
os.chdir(wd)

# Starts timer
start = time.time()

# Sets today's date for saving of dynamic file names
today = pd.Timestamp('today').strftime('%d-%m-%y')

'''
# Deletes any files already in working directory (sends to trash)
for filename in os.listdir():
    send2trash.send2trash(filename)
'''
print(str(os.listdir()) + '\n')

#####

# CREATE DATAFRAME WITH ALL TEST CONDITIONS
feature_extraction_methods = ['Binary',
                              'TF',
                              'TF-IDF',
                              ]
dimensionality_reduction_methods = ['T-SVD',
                                    'Thresholds',
                                    ]
tsvd_components = [100, 200, 400, 600, 800, 1600]
min_dfs = [2]
max_dfs = [0.85] * len(min_dfs)
cca_components = []
all_experimental_groups = doc_sim.experimental_groups_gen(feature_extraction_methods,
                                                          min_dfs,
                                                          max_dfs,
                                                          tsvd_components,
                                                          cca_components,
                                                          )

all_results = OrderedDict()
all_correct_links = []

#####

# Generates train/test datasets according to pre-defined tags
pm_corpus, web_corpus, full_corpus, corpora_links = ft_x.train_test_corpus()

# LOADING DOCUMENT CORPORA AND ASSOCIATED DATA FROM FILE

# Loads dictionary containing lengths of training/test corpora
corpus_lengths = pickle.load(open('test_train_corpus_lengths.pkl', 'rb'))

# Loads full combined PubMed and web corpus
full_web_pm_corpus = pd.read_pickle('full_web_pm_corpus_text.pkl')

# Loads TRAIN PubMed and web datasets
full_train_corpus = pd.read_pickle('all_train_datasets.pkl')
corpora_links_train = full_train_corpus['corpora_links_train']
web_corpus_train = full_train_corpus['web_corpus_train']
pm_corpus_train = full_train_corpus['pm_corpus_train']

# Loads TEST PubMed and web datasets
full_test_corpus = pd.read_pickle('all_test_datasets.pkl')
corpora_links_test = full_test_corpus['corpora_links_test']
web_corpus_test = full_test_corpus['web_corpus_test']
pm_corpus_test = full_test_corpus['pm_corpus_test']

# Loads full datasets
full_web_pm_matrix_order = pickle.load(open('full_web_pm_matrix_order.pkl', 'rb'))
full_web = full_web_pm_matrix_order['full_web']
full_pm = full_web_pm_matrix_order['full_pm']


for index, params_dict in enumerate(all_experimental_groups):
    print(params_dict)

    feature_extraction = params_dict['Feature Extraction']
    dimensionality_reduction = params_dict['Dimensionality Reduction']
    apply_cca = params_dict['CCA']
    tsvd_components = params_dict['T-SVD Components']
    cca_components = params_dict['CCA Components']
    min_df = params_dict['Min DF']
    max_df = params_dict['Max DF']

#####

# FEATURE EXTRACTION #
    if dimensionality_reduction == 'Thresholds':
        vocab_thresholds = ft_x.vocab_gen(pm_corpus,
                                          web_corpus,
                                          full_corpus,
                                          params_dict,
                                          )
        print('Length of thresholds vocab: %s' % len(vocab_thresholds))
        '''     
        # Represents web and PubMed documents in the vector space (generates feature matrices)
        ft_x.feature_representation(vocab_thresholds,
                                    full_corpus,
                                    params_dict,
                                    )
        '''
    elif (dimensionality_reduction == 'T-SVD') and\
            (min_df == all_experimental_groups[index - 1]['Min DF']) and\
            (max_df == all_experimental_groups[index - 1]['Max DF']) and \
            (feature_extraction == all_experimental_groups[index - 1]['Feature Extraction']):
        print('Vocabulary and feature matrix already generated for this experimental group')

    else:
        # Generates vocabulary lists used for feature representations
        # vocab_no_thresholds: All distinct words common to both PubMed and web corpora
        # vocab_thresholds: All distinct words common to both corpora, with document frequencies within thresholds
        vocab_no_thresholds = ft_x.vocab_gen(pm_corpus,
                                             web_corpus,
                                             full_corpus,
                                             params_dict,
                                             )
        print('Length of no thresholds vocab: %s' % len(vocab_no_thresholds))

        # Represents web and PubMed documents in the vector space (generates feature matrices)
        ft_x.feature_representation(vocab_no_thresholds,
                                    full_corpus,
                                    params_dict,
                                    )

    #####

    #  DOCUMENT SIMILARITY + PERFORMANCE MEASURES
    web_train, web_test, pm_train, pm_test = doc_sim.train_test_matrix(corpus_lengths,
                                                                       params_dict,
                                                                       )
    correct_links_ranks = doc_sim.cosine_sim(web_test,
                                             pm_test,
                                             web_corpus_test,
                                             pm_corpus_test,
                                             corpora_links_test,
                                             params_dict,
                                             )
    correct_links_ranks.to_pickle('%s_%s_%s_%s_%s_%s_%s_correct_link_ranks.pkl' %
                                  tuple([value for value in params_dict.values()]))
    all_correct_links.append(correct_links_ranks)

    results_dict = doc_sim.measures(correct_links_ranks,
                                    params_dict,
                                    )
    print('\n' + str(results_dict) + '\n')
    all_results['%s_%s_%s_%s_%s_%s_%s' % tuple([value for value in params_dict.values()])] = results_dict
    print('%s_%s_%s_%s_%s_%s_%s added to results list' %
          tuple([value for value in params_dict.values()]))

all_group_names = [key for key in params_dict.keys()]

results_df = pd.DataFrame.from_dict(all_results,
                                    columns=[key for key in results_dict.keys()],
                                    orient='Index',
                                    )

all_correct_links.to_pickle('ALL_CORRECT_RANKS.pkl')
list(results_df.index).to_pickle('ALL_GROUP_NAMES.pkl')

results_df.to_pickle('ALL_CCA_RESULTS.pkl')
results_df.to_csv('ALL_CCA_RESULTS.csv')

