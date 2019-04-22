"""
Uses the NCBI E-Utilities API to retrieve article content for all PMIDs supplied via
an input CSV file ('altmetric_pmids.csv'). Saves retrieved data to output CSV file.

Processing Steps:
1) Load list of PMIDs from file
2) Retrieve article attributes for each PMID using NCBI E-Utilitis API
3) Re-attempt to download any unsuccessful articles
4) Remove any duplicate PubMed articles from dataset
5) Save to file

"""

import os
import pandas as pd
import retrieval_cleansing_functions as prep
import glob


# Sets today's date for saving of dynamic file names
today = pd.Timestamp('today').strftime('%d-%m-%y')


#####


# PUBMED ARTICLE RETRIEVAL #
# DATASET IMPORT
# Imports original list of PubMed IDs
os.chdir('/Users/lizaharrison/PycharmProjects/web2pubmed')
pmid_df = pd.read_csv('altmetric_pmids.csv',
                      header=None,
                      usecols=[0],
                      names=['PMID'],
                      )
pmid_df.sort_values(by='PMID',
                    inplace=True,
                    )
pmid_df.reset_index(inplace=True,
                    drop=True,
                    )

print('No. PubMed articles for download: ' + str(len(pmid_df)))


#####


# API
# CRASH: eutils.exceptions.EutilsRequestError: Bad Request (400): error while forwarding to eutils
# crash = 277299
# print(pmid_df.iloc[crash + 1:])

# PUBMED ARTICLE RETRIEVAL (API)
# Accesses articles corresponding to each PMID via E-Utilities API
pm_data_1 = prep.eutils_from_df(pmid_df, 100, 'PubMed_Data_' + str(today) + '.csv')

print('Shape:\n'
      + str(pm_data_1.shape)
      + '\nColumns:\n'
      + str(pm_data_1.columns)
      )


#####


# UNSUCCESSFUL RETRIEVAL RE-ATTEMPT
# Reads all CSV files containing PubMed data into single DataFrame for analysis (all data fields)
# Specifies field names for PubMed article data and CSV files
fieldnames = ['PMID',
              'Year',
              'Title',
              'Abstract',
              'Authors',
              'Journal',
              'Volume',
              'Issue',
              'Pages',
              'DOI',
              'PMC', ]

all_files = glob.glob('*.csv')

print('Importing PubMed article data...')
pubmed_download_1 = pd.concat((pd.read_csv(file,
                                           skiprows=[0],
                                           header=None,
                                           names=fieldnames,
                                           encoding='utf-8',
                                           na_values='None',
                                           dtype={'PMID': int},
                                           )
                               for file in all_files
                               ),
                              ignore_index=True,
                              ).fillna('n/a')
print('Import Complete')

print('Shape:\n' + str(pubmed_download_1.shape))
print('Columns:\n' + str(pubmed_download_1.columns))

# Identifies any articles unable to be accessed via PubMed API by populating new boolean column ['Downloaded']
# Extracts unsuccessful PMIDs to new dataframe for second retrieval attempt
# Re-attempts to access these articles via E-Utilities API using pre-defined function
pmid_df['Downloaded'] = pmid_df['PMID'].isin(pubmed_download_1['PMID'])
print('No. Unsuccessful Downloads: ' + str(len(pmid_df.loc[~pmid_df['Downloaded']])))

retry = pmid_df.loc[pmid_df['Downloaded'] is False, ['PMID']]
retry.reset_index(inplace=True,
                  drop=True,
                  )

retry_df = prep.eutils_from_df(retry, 100, 'pm_data_retry_' + str(today) + '.csv')
print('Number of successful retries: '
      + str(len(retry_df))
      )

# Appends records successfully retrieved via retry attempt to full dataset
pubmed_download_2 = pubmed_download_1.append(retry_df)
pubmed_download_2.reset_index(inplace=True,
                              drop=True,
                              )


#####


# IMPORTS RETRIEVED DATA FROM FILES
# Imports all CSV files containing PubMed data into single DataFrame for analysis (all data fields)
# Specifies field names for PubMed article data and CSV files
fieldnames = ['PMID',
              'Year',
              'Title',
              'Abstract',
              'Authors',
              'Journal',
              'Volume',
              'Issue',
              'Pages',
              'DOI',
              'PMC', ]

all_files = glob.glob('PubMed_Data*.csv')

print('Importing PubMed article data...')
pm_data_1 = pd.concat((pd.read_csv(file,
                                   skiprows=[0],
                                   header=None,
                                   names=fieldnames,
                                   encoding='utf-8',
                                   na_values='None',
                                   dtype={'PMID': int}) for file in all_files),
                      ignore_index=True,
                      )
print('Import Complete')

print('Shape:\n' + str(pm_data_1.shape))
print('Columns:\n' + str(pm_data_1.columns))


#####


# DUPLICATE CLEANSING
# Identifies any articles fully duplicated in the PubMed dataset
# Removes repeats of header row (from CSV file import) and duplicate articles (from multiple download sessions)
dup_ids = pm_data_1.loc[pm_data_1.duplicated(subset='PMID',
                                             keep=False,
                                             )]
pm_data_v2 = pm_data_1.drop_duplicates(subset='PMID', )

print('\nNo. Duplicated PMIDs: '
      + str(len(dup_ids)),
      '\nFinal No. Downloaded PubMed articles: '
      + str(len(pm_data_v2)),
      '\nOriginal No. PMIDs: '
      + str(len(pmid_df)),
      '\nArticles not retrieved due to error: '
      + str(len(pmid_df) - len(pm_data_1))
      )


#####


# SAVE TO FILE
# Pickles dataframes for later cleansing
pd.to_pickle(pm_data_v2, str(today) + '_PubMed_Data_Original.pkl')
print('Save to *pkl complete')

# Saves dataframes to CSV for storage
pm_data_v2.to_csv(str(today) + '_PubMed_Data_Original.csv')
print('Save to *csv complete')
