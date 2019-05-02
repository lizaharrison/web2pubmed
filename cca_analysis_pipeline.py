"""
This module performs
"""
import config
import os
import time
import pickle
import pandas as pd
import document_similarity_functions as doc_sim
from collections import OrderedDict
import scipy

#####

# SET-UP
wd = config.working_dir
os.chdir(wd)

# Starts timer
start = time.time()

# Sets today's date for saving of dynamic file names
today = pd.Timestamp('today').strftime('%d-%m-%y')

#####

# CREATE DATAFRAME WITH ALL TEST CONDITIONS
feature_extraction_methods = ['TF-IDF']
dimensionality_reduction_methods = ['T-SVD']
tsvd_components = [100, 1600, 800, 400, 200]
tsvd_components.sort()
min_dfs = []
max_dfs = [] * len(min_dfs)
cca = [True]
cca_components = [50, 100, 200, 400, 800]
cca_components.sort()

all_experimental_groups = doc_sim.experimental_groups_gen(feature_extraction_methods,
                                                          min_dfs,
                                                          max_dfs,
                                                          tsvd_components,
                                                          cca_components,
                                                          )

[print(x) for x in all_experimental_groups]

all_results = OrderedDict()
all_correct_links = []

#####

# LOADING DOCUMENT CORPORA AND ASSOCIATED DATA FROM FILE
wd = '/Users/lizaharrison/PycharmProjects/Predicting_Papers_v1/Final_Datasets'
os.chdir(wd)

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

#####

wd = '/Users/lizaharrison/PycharmProjects/Predicting_Papers_v1/Results/CCA_Results_2'
os.chdir(wd)

all_cca_filenames = os.listdir()
all_cca_filenames.sort()

# all_cca_pm = [scipy.sparse.load_npz(filename) for filename in all_cca_filenames if filename.endswith('pm_test.npz')]
all_cca_pm_names = [filename for filename in all_cca_filenames if filename.endswith('pm_test.npz')]
# all_cca_web = [scipy.sparse.load_npz(filename) for filename in all_cca_filenames if filename.endswith('web_test.npz')]
all_cca_web_names = [filename for filename in all_cca_filenames if filename.endswith('web_test.npz')]

print([x for x in all_cca_pm_names])
print([x for x in all_cca_web_names])

for index, params_dict in enumerate(all_experimental_groups):
    print([params_dict])
    if params_dict['T-SVD Components'] == 1600 and\
            params_dict['CCA Components'] in [1600, 800]:
        pass
    else:
        web_file = [file for file in all_cca_web_names if file == ('TF-IDF_T-SVD_Applied_%s_cca_%s_web_test.npz' %
                                                                          (params_dict['T-SVD Components'],
                                                                           params_dict['CCA Components']
                                                                           ))]
        pm_file = [file for file in all_cca_pm_names if file.startswith('TF-IDF_T-SVD_Applied_%s_cca_%s_pm_test.npz' %
                                                                          (params_dict['T-SVD Components'],
                                                                           params_dict['CCA Components']
                                                                           ))]
        web_test = scipy.sparse.load_npz(web_file[0])
        pm_test = scipy.sparse.load_npz(pm_file[0])

        print(web_file)
        print(pm_file)

        correct_links_ranks = doc_sim.cosine_sim(web_test,
                                                 pm_test,
                                                 web_corpus_test,
                                                 pm_corpus_test,
                                                 corpora_links_test,
                                                 params_dict,
                                                 )
        correct_links_ranks.to_pickle('%s_%s_%s_%s_%s_%s_%s_correct_link_ranks.pkl' %
                                      tuple([value for value in params_dict.values()]))
        print(correct_links_ranks)
        all_correct_links.append(correct_links_ranks)

        results_dict = doc_sim.measures(correct_links_ranks,
                                        params_dict,
                                        )
        print('\n' + str(results_dict) + '\n')
        all_results['%s_%s_%s_%s_%s_%s_%s' % tuple([value for value in params_dict.values()])] = results_dict
        print('%s_%s_%s_%s_%s_%s_%s added to results list' %
              tuple([value for value in params_dict.values()]))

results_df = pd.DataFrame.from_dict(all_results,
                                    columns=[key for key in results_dict.keys()],
                                    orient='Index',
                                    )

pickle.dump(all_correct_links, open('ALL_CORRECT_RANKS.pkl', 'wb'))
pickle.dump(list(results_df.index), open('ALL_GROUP_NAMES.pkl', 'wb'))

results_df.to_pickle('ALL_CCA_RESULTS.pkl')
results_df.to_csv('ALL_CCA_RESULTS.csv')

all_groupnames = []
for group in all_experimental_groups:
    groupname = '%s_%s_%s_%s_%s_%s_%s' % tuple([value for value in params_dict.values()])
    all_groupnames.append(groupname)
