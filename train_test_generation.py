import config
import os
import pandas as pd
import database_functions as db_fnc
import numpy as np
from sqlalchemy import create_engine

#####

# SET-UP
wd = config.WORKING_DIR
os.chdir(wd)

# Sets today's date for saving of dynamic file names
today = pd.Timestamp('today').strftime('%d-%m-%y')

#####

# DATABASE DOWNLOAD
# Established database connection
cnxn = db_fnc.database_connect(db_fnc.db, db_fnc.host, db_fnc.user, db_fnc.password)
all_data = db_fnc.db_to_df(db_fnc.final_corpora_scripts, cnxn)

# Extracts each dataset (database table) from list
pm_corpus = all_data[0].set_index('pmid').sort_values('web_id')
web_corpus = all_data[1].set_index('web_id').sort_index()
corpora_links = all_data[2].set_index('web_id').sort_index()

#####

# GENERATION OF TRAINING AND TESTING DATASETS
# Generates training set by selecting half of the linked records in the sample set
# NOTE: PubMed training set contains only those articles with known links to the articles in the web training set
train_size = int(round(len(corpora_links) * 0.7, 0))
# train_size = int(round(len(corpora_links) * config.train_portion, 0))
test_size = len(corpora_links) - train_size
corpora_links_train = corpora_links.sample(n=train_size)

# Generates test set by selecting all other records in the sample set
# NOTE: PubMed test set contains both linked and unlinked articles [n = len(pm_corpus_1) - len(pm_corpus_train)]
corpora_links_test = corpora_links.loc[~corpora_links.index.isin(corpora_links_train.index)]
corpora_links['train_test'] = np.where(corpora_links.index.isin(corpora_links_train.index),
                                       'Train',
                                       'Test',
                                       )

#####

# DATABASE UPLOAD
# Uploads corpora links DataFrame with assigned Train/Test flag to database table
engine = create_engine('postgresql://liza:4Eiadb9@localhost:5432/predicting_papers_v1')

corpora_links.to_sql('final_links',
                     engine,
                     index=True,
                     if_exists='replace',
                     )

engine.dispose()

print('Corpus links table in database updated with Train/Test tag')
