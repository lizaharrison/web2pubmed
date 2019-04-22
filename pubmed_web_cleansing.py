"""
Cleanses and prepares webpage and PubMed article data to generate final set of 1:1 links
for analyses.

Processing Steps:
1) Retrieve data from database
2) Cleanse PubMed text & remove PubMed articles with text length below minimum threshold (100)
3) Cleanse webpage text & remove webpages with text length below minimum threshold (100)
4) Remove non-English webpages
5) Removal of highly-similar webpages - Longest common substring (LCS) analysis
6) Visualisation of webpage to PubMed article links
7) Selection of 1:1 PMID-URL links
8) Save final datasets to database

Database tables:
1) final_pubmed_corpus:
2) final_web_corpus:
3) pmid_web_links_full:
4) final_links:


"""

import pandas as pd
import database_functions as db_fnc
import retrieval_cleansing_functions as prep
from langdetect import detect
from sqlalchemy import create_engine
import numpy as np
import itertools as it
import pickle
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

#####

# DATASET DOWNLOAD FROM DATABASE
# Established database connection
cnxn = db_fnc.database_connect(db_fnc.db, db_fnc.host, db_fnc.user, db_fnc.password)
all_data = db_fnc.db_to_df(db_fnc.original_dataset_scripts, cnxn)

# Extracts each dataset (database table) from list
pmid_og = all_data[0]
pmid_og.columns = ['pmid']

pm_data_1 = all_data[1]
pm_data_1.columns = [col.lower() for col in pm_data_1.columns.values]
pm_data_1.set_index('pmid',
                    inplace=True,
                    )


#####


# GENERATION OF PUBMED CORPUS TEXT
# Concatenates PubMed article titles and abstracts into single field
# Forms PubMed article corpus for vectorization
pm_data_1.fillna('', inplace=True)
pm_data_1['corpus_text'] = pd.Series(pm_data_1['title'] + ': ' + pm_data_1['abstract'])


#####


# REMOVAL OF PUBMED ARTICLES W/ MISSING TEXT
# Standardises known <null> values for identification of missing title and abstract text
pm_data_1.replace([r'^n/a$',
                   r'^\s+$',
                   ],
                  np.NaN,
                  regex=True,
                  inplace=True,
                  )

# Creates column containing length of corpus text
pm_data_1['text_length'] = pm_data_1['corpus_text'].apply(lambda x: len(x.split()))

# Identifies any articles without title or abstract text
no_text = pm_data_1.loc[(pm_data_1['text_length'] == 0)]

# Removes any records with less than 100 words of text for analysis
min_len_threshold = 100
pm_data_2 = pm_data_1.drop((pm_data_1.index[(pm_data_1['text_length'] < min_len_threshold)]))

print('No. PubMed articles w/ missing titles AND abstracts: '
      + str(len(no_text)) +
      '\nNo. PubMed articles w/ less than 100 words of text for analysis: '
      + str(len(pm_data_1.loc[pm_data_1['text_length'] < min_len_threshold]))
      )

print('No. PubMed articles remaining with more than 100 words of text for analysis: '
      + str(len(pm_data_2)))


#####


# CREATION OF WEB DATASET

web_data_1 = all_data[2]

# Identifies web records with PMIDs no longer in PubMed dataset following first cleansing steps
missing_pmids = web_data_1.loc[~web_data_1['pmid'].isin(pm_data_2.index)]
# Removes web records with missing PMIDs
web_data_2 = web_data_1.drop(missing_pmids.index.values,
                             )
web_data_1.reset_index(inplace=True,
                       drop=False,
                       )
web_data_1.columns = ['web_id', 'processed_content', 'url', 'pmid']

'''pm_web_links = all_data[3].reset_index(drop=False)
pm_web_links.columns = ['link_id', 'pmid', 'url']
'''
print('Pubmed corpus fields: ' + str(pm_data_2.columns.values))
print('Web corpus fields: ' + str(web_data_1.columns.values))
# print('Web to PubMed links: ' + str(pm_web_links.columns.values))


#####


# CLEANSING OF WEB ARTICLE TEXT
# Removes TITLE and TEXT headings from web text data and replaces with delimiter not found in any records
processed_content_1 = web_data_2.replace(
    {'processed_content':
        {'TITLE: ': '',
         r'\n': '',
         'TEXT: ': r'{-}',
         }
     }, regex=True).loc[:, 'processed_content']

# Replaces whitespace character strings with single space
web_data_2['processed_content'] = web_data_2['processed_content'].apply(lambda x: ' '.join(x.split()))


# Creates new columns for TITLE and TEXT for analysis of content
web_data_2[['title', 'content']] = processed_content_1.str.split(r'{-}',
                                                                 n=1,
                                                                 expand=True,
                                                                 )
print('Checking successful delmiiting: \n'
      + str(web_data_2.loc[0:10, ['title',
                                  'content',
                                  ]]
            )
      )

# Strips trailing whitespace from all article text fields
web_data_2['processed_content'] = [str(text).strip(' ') for text in web_data_2['processed_content']]
web_data_2['title'] = [str(title).strip(' ') for title in web_data_2['title']]
web_data_2['content'] = [str(content).strip(' ') for content in web_data_2['content']]


#####


# REMOVAL OF WEB RECORDS W/ MISSING TEXT
# Identification of web articles with blank title or content (article text) fields
web_data_2 = web_data_2.assign(
    title_nan=(web_data_2['title'].isnull()) |
              (web_data_2['title'].isin(['', '0', ' ', 'None']))).assign(
    content_nan=(web_data_2['content'].isnull()) |
                (web_data_2['content'].isin(['', '0', ' ', 'None'])))

# Confirms successful tagging of missing titles/content
print(web_data_2.loc[web_data_2['title_nan'], ['title', ]],
      web_data_2.loc[web_data_2['content_nan'], ['content', ]]
      )

print('No. articles w/ missing titles: '
      + str(len(web_data_2.loc[web_data_2['title_nan']])),
      '\nNo. articles w/ missing content: '
      + str(len(web_data_2.loc[web_data_2['content_nan']])),
      '\nNo. articles w/ missing title AND content: '
      + str(len(web_data_2.loc[(web_data_2['title_nan']) &
                               (web_data_2['content_nan'])]))
      )

# Removal of web articles with blank title and content (article text) fields (absence of text features)
web_data_3 = web_data_2.drop((web_data_2.index[(web_data_2['title_nan']) &
                                               (web_data_2['content_nan'])]
                              ))
print('Web Data 3\n'
      '(Rows, columns)\n'
      + str(web_data_3.shape)
      )


#####


# REMOVAL OF WEB RECORDS W/ TEXT BELOW MINIMUM THRESHOLD LENGTH
# Identification of records with less than 100 words of text
web_data_3['text_lengths'] = web_data_3['processed_content'].apply(lambda x: len(x.split()))
print('There are %s web records with less than 100 words in corpus text.' %
      web_data_3.loc[web_data_3['text_lengths'] < min_len_threshold])

web_data_4 = web_data_3.drop((web_data_3.index[web_data_3['text_lengths'] < min_len_threshold]))
print('Web Data 4\n'
      '(Rows, columns)\n'
      + str(web_data_3.shape)
      )


#####


# REMOVAL OF FOREIGN LANGUAGE ARTICLES
# https://cloud.google.com/translate/docs/detecting-language#translate_detect_language-python
# https://github.com/ssut/py-googletrans

# Concatenates cleansed web article titles and abstracts into single field
web_data_4['corpus_text'] = pd.Series(web_data_4['title'] + ': ' + web_data_4['content'])

# Extracts first 200 characters of each web article for language detection
word_30 = [' '.join(string.split()[0:30]) for string in web_data_4['corpus_text']]

foreign_lang = []

for index, string in enumerate(word_30):
    language = detect(string)
    if language != 'en':
        foreign_lang.append(index)

web_foreign = web_data_4.iloc[foreign_lang]
web_data_5 = web_data_4.drop(web_foreign.index)

print('Removal of foreign language articles complete')


#####


# CLEANSING OF DUPLICATE WEB RECORDS
# Identifies and removes exact duplicate records
# web_data_5: All articles with unique combinations of URL and PMID
dup_content_pmid = web_data_5.loc[web_data_5.duplicated(subset=['content',
                                                                'pmid',
                                                                ],
                                                        keep=False,
                                                        )]

print(str(len(dup_content_pmid))
      + ' article records are duplicated in at least content text and PMID.'
      + ' In total these duplicated records appear '
      + str(len(web_data_5.loc[web_data_5.duplicated(subset=['content',
                                                             'pmid',
                                                             ],
                                                     keep=False,)]))
      + ' times in the web article dataset.'
      )

web_data_6 = web_data_5.drop_duplicates(subset=['content',
                                                'pmid',
                                                ],
                                        keep='first',
                                        )
web_data_6 = web_data_6.drop_duplicates(subset=['title',
                                                'pmid'],
                                        keep='first',
                                        )

print('Web Data 6\n'
      + '(Rows, columns)\n'
      + str(web_data_6.shape)
      )

web_data_6['pmid_count'] = pd.DataFrame(web_data_6.groupby('pmid')['pmid'].transform('size'))
web_data_6['url_count'] = pd.DataFrame(web_data_6.groupby('final_url')['final_url'].transform('size'))

dup_pmid_subset = web_data_6.loc[web_data_6['pmid_count'] > 1].sort_values(['pmid', 'corpus_text'])
# dup_url_subset = web_data_6.loc[web_data_6['url_count'] > 1].sort_values(['url', 'corpus_text'])
dup_pmid_grouped = dup_pmid_subset.groupby('pmid')
# dup_url_grouped = dup_pmid_subset[0:100].groupby('url')


lcs_dict = {}

# For all distinct pair of web ids that share the same PMID, calculates the longest common substring
for group_name, group in web_data_6.groupby('pmid'):
    pmid = group_name
    corpus_text_all = group['corpus_text']
    counter = 0
    pairs = it.permutations(corpus_text_all.index, 2)
    min_similarity = 50
    i = 0

    for pair in pairs:
        index_1, index_2 = pair
        str1 = corpus_text_all[index_1]
        str2 = corpus_text_all[index_2]
        i += 1

        if ('%s-%s' % (index_2, index_1)) in set(lcs_dict.keys()):
            continue

        else:
            # Calculates LCS
            lcs = prep.lcs_algorithm(str1, str2)

        # Calculates percentage of text shared between two articles
        if len(lcs) > 0:
            pct = (len(lcs[0]) / min(len(str1), len(str2))) * 100

        # Selects the web ids of those records with at least 50% of common text
        if pct > min_similarity:
            print('%s-%s Longest common substring > min threshold' % (index_1, index_2))
            lcs_dict['%s-%s' % (index_1, index_2)] = [pct, str1, str2, pmid]

# Saves to dataframe for analysis
print('Longest common substring analysis of all records complete')
lcs_df = pd.DataFrame.from_dict(lcs_dict).transpose()
lcs_df.columns = ['%age_common',
                  'corpus_text_1',
                  'corpus_text_2',
                  'pmid'
                  ]
lcs_df.reset_index(drop=False,
                   inplace=True,
                   )
lcs_df[['article_id_1', 'article_id_2']] = lcs_df['index'].str.split('-',
                                                                     expand=True
                                                                     )

lcs_index = lcs_df['index']

lcs_df.drop('index',
            inplace=True,
            axis=1,
            )

lcs_df_reverse = lcs_df.loc[:, ['%age_common',
                                'corpus_text_2',
                                'corpus_text_1',
                                'pmid',
                                'article_id_2',
                                'article_id_1',
                                ]]
lcs_df_reverse.columns = ['%age_common',
                          'corpus_text_1',
                          'corpus_text_2',
                          'pmid',
                          'article_id_1',
                          'article_id_2',
                          ]
lcs_df_2 = lcs_df.append(lcs_df_reverse)
lcs_df_2 = lcs_df_2.loc[:, ['article_id_1',
                            'article_id_2',
                            'corpus_text_1',
                            'corpus_text_2',
                            '%age_common',
                            'pmid'
                            ]]

lcs_df_2.drop_duplicates(['article_id_1',
                          'article_id_2',
                          ],
                         inplace=True,
                         )


group_ids = pd.factorize(lcs_df['pmid'])
lcs_df['group_id'] = group_ids[0]
lcs_df.set_index(['group_id'], inplace=True)
lcs_df.reset_index(inplace=True)
lcs_df.sort_values(['group_id',
                    'article_id_1',
                    'article_id_2',
                    ],
                   inplace=True,
                   )

pickle.dump(lcs_df, open('lcs_dataframe.pkl', 'wb'))

lcs_df_2 = pd.read_pickle('lcs_dataframe.pkl')

lcs_keep = []
lcs_drop = []

for pmid, group in lcs_df_2.groupby('pmid'):
    group = group.drop_duplicates(['article_id_2'])
    ids = group['article_id_1'].append(group['article_id_2']).drop_duplicates()

    if any(int(article_id) in lcs_keep for article_id in ids.values):
        lcs_drop.extend([int(article_id) for article_id in ids if article_id not in lcs_drop])

    else:
        keep = ids.sample(1).values
        # corpus_text = group['corpus_text_2']
        # lengths = [len(text) for text in corpus_text.values]
        # keep = ids.iloc[np.argmax(lengths)]

        if keep not in lcs_keep:
            lcs_keep.extend(keep)
            lcs_drop.extend([int(article_id) for article_id in ids if article_id is not keep])

keeping = web_data_6.loc[web_data_6.index.isin(lcs_keep), ['corpus_text', 'pmid']]

web_data_7 = web_data_6.drop(lcs_drop)
web_data_7.columns = ['processed_content', 'url', 'pmid', 'title', 'content',
                      'title_nan', 'content_nan', 'text_lengths', 'corpus_text',
                      'pmid_count', 'url_count',
                      ]

#####

# GENERATION OF FINAL WEB CORPUS
# Identifies distinct web articles in the dataset, independent of the PubMed article to which they have been linked
# Many web articles are mapped to more than one PMID, and as such appear more than once in the full dataset

# Identifes web records with duplicate (title and article content) independent of URL and PMID
dup_text_only = web_data_7.loc[web_data_7.duplicated(subset=['corpus_text',
                                                             'title',
                                                             'content',
                                                             ],
                                                     keep='first',
                                                     )]
dup_text_url = web_data_7.loc[web_data_7.duplicated(subset=['corpus_text',
                                                            'title',
                                                            'content',
                                                            'url',
                                                            ],
                                                    keep='first',
                                                    )]

print('''There are %s distinct web pages (URLs) in the web dataset.
      \nThere are %s distinct articles (title and content text) in the web dataset.
      \n%s distinct PubMed articles are represented in the web dataset.''' %
      (len(web_data_7.loc[:, 'url'].unique()),
       len(web_data_7) - len(dup_text_only),
       len(web_data_7.loc[:, 'pmid'].unique())))

print(str(len(dup_text_only)) +
      ' articles appear more than once in the dataset (independent of PMID and URL)\n'
      + str(len(dup_text_url)) +
      ' articles appear more than once in the dataset (independent of PMID only)')

# Compiles two datasets for web articles:
# web_corpus_db: Distinct web articles only - based on title/content text and URL. Independent of linked PMID)
web_corpus_db = web_data_7.loc[:, ['corpus_text',
                                   'url',
                                   'title',
                                   'content',
                                   'processed_content',
                                   ]].drop_duplicates(subset=['corpus_text',
                                                              'title',
                                                              'content',
                                                              ],
                                                      keep='first',
                                                      )
web_corpus_db.columns = ['corpus_text',
                         'final_url',
                         'title',
                         'content',
                         'original',
                         ]
print('web_corpus_db columns: '
      + str(web_corpus_db.columns))

# web_data_w_links: All web records remaining in the cleansed dataset, including:
# - Data on links to PubMed articles
# - Articles with duplicate content but published on multiple web pages
web_corpus_w_links = web_data_7.loc[:, ['pmid',
                                        'corpus_text',
                                        'url',
                                        'title',
                                        'content',
                                        ]]
web_corpus_w_links.columns = ['pmid',
                              'corpus_text',
                              'url',
                              'title',
                              'content',
                              ]
print('web_corpus_w_links columns: '
      + str(web_corpus_w_links.columns))

print('No. distinct articles in web corpus: '
      + str(len(web_corpus_db))
      + '\nNo. web-PubMed article links: '
      + str(len(web_corpus_w_links))
      )

#####

# GENERATION OF FINAL PUBMED CORPUS

final_pm_corpus = pm_data_2.loc[:, ['corpus_text',
                                    'title',
                                    'abstract',
                                    'year',
                                    'authors',
                                    'journal',
                                    'volume',
                                    'issue',
                                    'pages',
                                    'doi',
                                    'pmc',
                                    ]]

print('PubMed corpus shape:\n'
      + str(final_pm_corpus.shape)
      )

#####

# GENERATION OF FINAL 1:1 PUBMED TO WEB LINKS DATASET
'''
LOGIC:
Generate a set of 1:1 PMID to URL links.

pmid_1_url_1: PMID only appears once in dataset, against a single URL also only appearing once in the dataset
pmid_many_url_1: URL appears only once in dataset, but it is linked to the same PMID as > 1
pmid_1_url_many: URL appears multiple times, but in all cases is
pmid_many_url_many:

1) Get all existing 1:1 records (1 PMID associated with only 1 URL, that URL only associated with that PMID)
2) For PMIDs with more than one web article, randomly pick one to include in dataset OR pick longest one
Each PMID should appear in the dataset only once, against the longest possible web article
'''

# Compiles all web articles that appear only once in the dataset
# Web records with linked PMIDs also appearing only once in dataset (1 : 1 relationship)
pmid_1_web_1 = web_data_7.loc[(web_data_7['pmid_count'] == 1) & (web_data_7['url_count'] == 1)]

# Web records for which one of the linked PMIDs only appears once in the dataset
pmid_1_web_many = web_data_7.loc[(web_data_7['pmid_count'] == 1) & (web_data_7['url_count'] > 1)]

# Web records that appear only once in the dataset, but which are linked to a PMID linked to other web records
pmid_many_web_1 = web_data_7.loc[(web_data_7['pmid_count'] > 1) & (web_data_7['url_count'] == 1)]

pmid_many_web_many = web_data_7.loc[(web_data_7['pmid_count'] > 1) & (web_data_7['url_count'] > 1)]


# VISUALISATION OF LINKS

links_1 = web_data_7.set_index('url')
links_1 = links_1.loc[:, 'pmid']

pmid_per_web = links_1.value_counts()
pmid_count_freq = pmid_per_web.value_counts()
pmid_count_freq.sort_index(inplace=True)

links_2 = web_data_7.set_index('pmid')
links_2 = links_2.loc[:, 'url']

web_per_pmid = links_2.value_counts()
web_count_freq = web_per_pmid.value_counts()
web_count_freq.sort_index(inplace=True)

fig = plt.figure(figsize=(11, 8))
fig.suptitle('Representation of PMIDs in web corpus')
fig, ax = plt.subplots()
ax.scatter(pmid_count_freq.index,
           pmid_count_freq,
           c='#1B4537',
           marker='o',
           s=4,
           )
ax.grid(True)
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style='plain')
plt.xlabel('Number of webpages linked to each PubMed article')
plt.ylabel('Number of PubMed articles per webpage count')
matplotlib.rcParams.update({'font.size': 13})

plt.savefig('Figure_3.png',
            dpi='figure',
            )

plt.show()


#####


# PUBMED ARTICLE AND WEB ARTICLE ONLY HAVE ONE REPORTED LINK
final_web_corpus_1 = pmid_1_web_1

final_web_corpus_2 = prep.select_web_records(pmid_1_web_many, final_web_corpus_1)
final_web_corpus_3 = prep.select_web_records(pmid_many_web_1, final_web_corpus_2)
final_web_corpus_4 = prep.select_web_records(pmid_many_web_many, final_web_corpus_3)

# Checks for duplicates
print('Duplicated URLs in final dataset: %s' % final_web_corpus_4.loc[final_web_corpus_4.duplicated('url')])
print('Duplicated PMIDs in final dataset: %s' % final_web_corpus_4.loc[final_web_corpus_4.duplicated('pmid')])
print('%s records in final web corpus' % len(final_web_corpus_4))


final_web_corpus_5 = final_web_corpus_4.drop(['processed_content',
                                              'title_nan',
                                              'content_nan',
                                              'text_lengths',
                                              'pmid_count',
                                              'url_count',
                                              ],
                                             axis=1,
                                             )

print(final_web_corpus_5.shape)

# Addition of web ids to PubMed corpus for those 1:1 links to be used for training
corpora_links = final_web_corpus_5.loc[:, ['web_id',
                                           'pmid',
                                           ]].reset_index(drop=True)
corpora_links.columns = ['web_id',
                         'pmid',
                         ]
corpora_links.astype(int)
corpora_links.sort_values('web_id',
                          inplace=True,
                          )

final_pm_corpus = final_pm_corpus.merge(corpora_links,
                                        left_on='pmid',
                                        right_on='pmid',
                                        how='left',
                                        )
print(final_pm_corpus.columns)

corpora_links['train_test'] = np.NaN


#####


# UPLOAD FINAL CORPORA TO DB
# Establishes connection to predicting_papers_v1 database using sqlalchemy

engine = create_engine('postgresql://liza:4Eiadb9@localhost:5432/predicting_papers_v1')

final_pm_corpus.to_sql('final_pubmed_corpus',
                       engine,
                       if_exists='replace',
                       index=False,
                       )

final_web_corpus_5.to_sql('final_web_corpus',
                          engine,
                          if_exists='replace',
                          index=False,
                          )
corpora_links.to_sql('final_links',
                     engine,
                     index=False,
                     if_exists='replace',
                     )

engine.dispose()

print('All datasets uploaded to database')
