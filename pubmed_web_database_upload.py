"""
This script inserts original PMID, PubMed article and web article data into pre-existing database tables.
Also inserted into an additional database table are the orignal known relationships between these
web and PubMed articles.

Web and PubMed article datasets are imported from Pickle files (*.pkl) and loaded into DataFrame
objects. DataFrame columns order is assumed to match that of the database tables into which each dataset is
to be inserted.

Database tables:
1) vaccine_pmids_original: Original list of PMIDs corresponding to articles retrieved by keyword search "vaccines".
    > Corresponding file - 'altmetric_pmids.csv'
2) web_data_full: Full set of web page data with known links to PMIDs in vaccine_pmids_original, as scraped using
    URLs provided by Altmetric.
    > Corresponding file - 'select_w_processed_content__w_final_url_.tsv'
3) pubmed_data_full: Full set of data retrieved by passing PMIDs in vaccine_pmid_original to E-Utilities API.
    Successfully downloaded articles include fields PMID, year, title, abstract, journal, volume, issue, pages, doi.
    This dataset has been de-duplicated by pmid, and had all articles with no text (no title or abstract) content
    removed.
    > Corresponding file - Combined 'pm_data_*.csv' + 'pm_data_retry.csv' files

"""

import os
import pandas as pd
import glob
from sqlalchemy import create_engine
import database_functions as db_fnc


# Sets today's date for saving of dynamic file names
today = pd.Timestamp('today').strftime('%d-%m-%y')

os.chdir('/Users/lizaharrison/PycharmProjects/Predicting_Papers_v1/Full Datasets/')


#####


# DATASET IMPORT FROM FILES
# Imports original list of vaccine-related PMIDs from file
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
print('No. PubMed IDs relating to vaccines: ' + str(len(pmid_df)))

# Imports PubMed article data for these PMIDs retrieved via E-Utilities API
pubmed_data = pd.read_pickle(glob.glob('*_PubMed_Data_Original.pkl')[0])
print('No. PubMed articles successfully retrieved: ' + str(len(pubmed_data)))

# Imports web article data for web pages with known links to these PMIDs (Source: Altmetric)
# Encodes data to UTF-8 prior to saving as DataFrame
with open('select_w_processed_content__w_final_url_.tsv',
          'r',
          encoding='utf-8',
          errors='ignore',
          ) as tsvfile:
    web_data = pd.read_csv(tsvfile,
                           delimiter='\t',
                           usecols=[0, 1, 2],
                           )
web_data.sort_values(['pmid',
                      'processed_content',
                      'final_url'
                      ]).reset_index(inplace=True)
print('No. web articles with known links to vaccine-related PubMed research: ' + str(len(web_data)))

web_pm_links = web_data.loc[:, ['pmid', 'final_url']].sort_values('pmid')
print('No. URL-PMID pairings in Altmetric data: ' + str(len(web_pm_links)))


#####


# DATABASE UPLOAD
# Establishes connection to predicting_papers_v1 database using sqlalchemy
engine = create_engine('postgresql://liza:4Eiadb9@localhost:5432/predicting_papers_v1')

pmid_df.to_sql('vaccine_pmids_original',
               engine,
               index=False,
               if_exists='replace')

web_data.to_sql('web_data_full',
                engine,
                index=False,
                if_exists='replace',
                )
print('Insertion of web articles into database complete')

pubmed_data.to_sql('pubmed_data_full',
                   engine,
                   index=False,
                   if_exists='replace',
                   )
print('Insertion of PubMed articles into database complete')

web_pm_links.to_sql('pmid_web_links_full',
                    engine,
                    index=False,
                    if_exists='replace',
                    )
print('Insertion of PubMed to web links into database complete')


#####


# GENERATION OF PUBMED TO WEB LINKS DATASET
scripts = ['SELECT * FROM web_data_full']

cnxn = db_fnc.database_connect(db_fnc.db,
                               db_fnc.host,
                               db_fnc.user,
                               db_fnc.password,
                               )
web_data_db = db_fnc.db_to_df(scripts,
                              cnxn,
                              )
print(web_data_db.columns)

web_pm_links = web_data_db.loc['']

web_pm_links.to_sql('pmid_web',
                    engine,
                    if_exists='append',
                    )
print('Insertion of PubMed-Web links into database complete')

# Establishes connection to predicting_papers_v1 database using psycopg2
cnxn = db_fnc.database_connect(db_fnc.db,
                               db_fnc.host,
                               db_fnc.user,
                               db_fnc.password,
                               )

print('Web-PubMed links:\n'
      + str(web_pm_links.shape)
      + '\n'
      + str(web_pm_links.columns)
      )


'''
SAMPLE CODE FOR DATABASE QUERY EXECUTION
sql = 'SELECT * FROM pmid_web'

cursor.execute(sql)
result_all = cursor.fetchall()
result_one = cursor.fetchone()
cursor.commit()
cursor.close()
'''
