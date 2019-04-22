"""
Author: Eliza Harrison

This module contains the function used to retrieve PubMed article metadata from the PubMed
database using the NCBI E-Utilities API, as well as all functions used for the cleansing and
preparation of PubMed and webpage data.

"""

import csv
import eutils
import requests
import lxml
import pandas as pd
import itertools
import re
import time


# E-UTILITIES PUBMED RETRIEVAL FUNCTION
def eutils_from_df(input_df, chunksize, output_csv):
    """
    Retrieves and saves PubMed article content from PubMed via E-Utilities API to CSV file
    for set of PMIDs contained within Pandas Dataframe.

    Args:
        input_df: object name for Dataframe containing PMIDs of interest
        chunksize: number of PMIDs to pass to API
        output_csv: filename for CSV file to which article content will be saved

    Returns:
        CSV file with rows pertaining to article content for each PMID in input_csv.
        Columns correspond to fields retrieved via efetch client:
            'PMID', 'Year', 'Title', 'Abstract', 'Authors', 'Journal', 'Volume', 'Issue',
            'Pages', 'DOI', 'PMC'
        List and dataframe containing all PubMed article data successfully retrieved from database
    """

    # Specifies names for output csv column headers
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
                  'PMC',
                  ]

    # Creates generator object containing each row in the input dataframe
    pm_chunks_gen = (input_df[i:i + chunksize] for i in range(0, len(input_df), chunksize))

    # Initialises empty list for compilation of article dictionaries into single container
    pm_article_list = []

    # Initialise eutils client to access NCBI E-Utilities API
    ec = eutils.Client()

    # Open CSV file to which each PubMed IDs downloaded data appended as a new row with specified column names
    with open(output_csv, 'a') as datafile:
        writer = csv.DictWriter(datafile,
                                fieldnames=fieldnames,
                                )
        writer.writeheader()

        # Converts each chunk of PubMed IDs from dataframe to list
        for chunk_count, chunk in zip(range(0, len(input_df)), pm_chunks_gen):
            try:
                index_list = list(chunk.index.values)
                chunk_list = list(chunk['PMID'])
                print('Chunk No. ' + str(chunk_count))

                # Passes chunk of PubMed IDs to E-Utilities API
                # Returns iterator object containing key data for each PubMed ID
                pm_article_set = iter(ec.efetch(db='pubmed',
                                                id=chunk_list,
                                                )
                                      )

                # Assigns each PubMed ID an index value
                # Iterates over pm_article_set to access data for each individual PubMed ID
                for id_index, id_value in enumerate(chunk_list):
                    print(index_list[id_index], id_value)
                    try:
                        # For each PMID index/value pair, iterates through article set
                        # Aggregates key article attributes for each PubMed ID into dictionary
                        pm_article = next(pm_article_set)
                        pm_article_content = dict(
                            PMID=str(pm_article.pmid),
                            Year=str(pm_article.year),
                            Title=str(pm_article.title),
                            Abstract=str(pm_article.abstract),
                            Authors=str(pm_article.authors),
                            Journal=str(pm_article.jrnl),
                            Volume=str(pm_article.volume),
                            Issue=str(pm_article.issue),
                            Pages=str(pm_article.pages),
                            DOI=str(pm_article.doi),
                            PMC=str(pm_article.pmc),
                            )

                        print(pm_article_content)
                        print(pm_article.pmid + ' - Download from Enterez complete')

                        # Saves dictionary as new item in list for later construction of dataframe
                        pm_article_list.append(pm_article_content)
                        print(pm_article.pmid + ' - Save to list complete')

                        # Writes dictionary to new row of csv file for future reference
                        writer.writerow(pm_article_content)
                        print(pm_article.pmid + ' - Write Data to CSV Complete')

                    # Except statements for content errors
                    except (StopIteration,
                            TypeError,
                            NameError,
                            ValueError,
                            lxml.etree.XMLSyntaxError,
                            eutils.exceptions.EutilsNCBIError,
                            ) as e1:
                        print('Error: ' + str(e1))
                        continue
                    # Except statements for network/connection errors
                    except(TimeoutError,
                           RuntimeError,
                           ConnectionError,
                           ConnectionResetError,
                           eutils.exceptions.EutilsRequestError,
                           requests.exceptions.ConnectionError,
                           ) as e2:
                        print('Error: ' + str(e2))
                        time.sleep(10)
                        continue

            except StopIteration:
                print('All downloads complete')
                break

    # Save list of dictionaries to dataframe & write to CSV file
    pm_article_df = pd.DataFrame.from_records(pm_article_list,
                                              columns=fieldnames,
                                              )
    print('Save to DataFrame complete')
    datafile.close()
    return pm_article_df


# LONGEST COMMON SUBSTRING ANALYSIS FUNCTIONS
def lcs_algorithm(str1, str2):
    """
    Extracts the longest common substring (words) between two strings.

    Args:
        str1: Input string 1
        str2: Input string 2

    Returns:
        lcs_set: The longest common substring shared between the two input strings.


   SOURCE: Code adapted from
        https://www.bogotobogo.com/python/python_longest_common_substring_lcs_algorithm_generalized_suffix_tree.php
    NOTE: Commented out sections correspond to code that allows for saving of more than one
        longest common substring (where len(lcs) is the same).
    """
    # str1_words = ''.join(str1.split())
    # str2_words = ''.join(str2.split())

    # Removes punctuation from string to prevent premature termination of longest common substring
    str1 = re.sub(r'[^\w\s]', '', str1)
    str2 = re.sub(r'[^\w\s]', '', str2)

    # Splits string into tuple of words, to compute lcs by word (vs character)
    str1_words = tuple(word for word in str1.lower().split())
    str2_words = tuple(word for word in str2.lower().split())

    m = len(str1_words)
    n = len(str2_words)

    matrix = [[0] * (n + 1) for i in range(m + 1)]

    longest = 0
    lcs_set = set()

    for i in range(m):

        for j in range(n):

            if str1_words[i] == str2_words[j]:
                x = matrix[i][j] + 1
                matrix[i+1][j+1] = x

                if x > longest:
                    longest = x
                    lcs_set = set()
                    lcs_set.add(str1_words[i - x + 1: i + 1])

                else:
                    pass

    lcs = [' '.join(tup) for tup in lcs_set]

    return lcs


def lcs_analysis(series, min_similarity):
    """
    Calculates and extracts longest common substring (lcs) between pairs of strings.

    Args:
        series:
        min_similarity:

    Returns:
        lcs_df:

    NOTE: Commented out sections correspond to code that allows for saving of more than one
        longest common substring (where len(lcs) is the same).
    """
    t0 = time.time()
    print('Beginning Longest Common Substring analysis...')

    # Removes all whitespace from series values to reduce size
    # series = series.apply(lambda x: ''.join(x.split()))  # This is stopping it from working on the title dataset

    # Generates all distinct pairs of series values
    article_pairs = itertools.combinations(series.index, 2)

    # Initialises empty list in which indices, lcs and %age match can be stored
    lcs_list = [[], [], [], []]

    for pair in article_pairs:
        index_1, index_2 = pair

        if index_1 != index_2:
            str1 = series[index_1]
            str2 = series[index_2]
            lcs = lcs_algorithm(str1, str2)

            if len(lcs) > 0:
                pct = (len(lcs[0]) / max(len(str1), len(str2))) * 100
                # pct = len(lcs[0]) / max(len(str1), len(str2)) * 100

                if pct > min_similarity:
                    print('%s - %s Longest common substring > min threshold' % (index_1, index_2))
                    lcs_list[0].append(index_1)
                    lcs_list[1].append(index_2)
                    lcs_list[2].append(lcs)
                    lcs_list[3].append(pct)

                else:
                    pass
            else:
                pass
        else:
            pass

    print('Longest common substring analysis of all records complete')
    lcs_df = pd.DataFrame(lcs_list).transpose()
    lcs_df.columns = ['article id 1', 'article id 2', 'lcs', '%age common']

    t1 = time.time()
    elapsed = round(t1 - t0, 2)
    print('%s seconds / %s minutes elapsed/n' % (elapsed, elapsed / 60))

    return lcs_df


# SELECTION OF 1:1 PMID-URL LINKS
def select_web_records(for_selection, full_corpus):
    subset = for_selection.loc[(~for_selection['pmid'].isin(full_corpus['pmid'].values))
                               & (~for_selection['url'].isin(full_corpus['url'].values))]

    selected_web_ids = []
    selected_urls = []

    for pmid in subset.loc[:, 'pmid'].unique():
        web_ids = subset.loc[subset['pmid'] == pmid].index.values
        urls = subset.loc[subset['pmid'] == pmid, 'url'].values
        corpus_text = subset.loc[subset['pmid'] == pmid, 'corpus_text'].values
        text_lengths = [len(text) for text in corpus_text]
        not_in_corpus = True

        while not_in_corpus is True:
            if len(web_ids) > 1:
                i = text_lengths.index(max(text_lengths))  # for selecting longest web article
                # i = random.randint(0, len(web_ids) - 1)  # for selecting random url
            elif len(web_ids) == 1:
                i = 0
            else:
                not_in_corpus = False
                break

            selected_id = web_ids[i]
            selected_url = urls[i]
            selected_corpus_text = corpus_text[i]

            if selected_url in selected_urls or \
                    selected_url in full_corpus['url'].values:
                web_ids = [x for x in web_ids if x != selected_id]
                urls = [x for x in urls if x != selected_url]
                corpus_text = [x for x in corpus_text if x != selected_corpus_text]
                text_lengths = [len(text) for text in corpus_text]
                not_in_corpus = True

            else:
                selected_web_ids.append(selected_id)
                selected_urls.append(selected_url)
                not_in_corpus = False
                print('Web record %s added to final corpus (%s)' % (selected_id, selected_url))

    full_corpus_updated = full_corpus.append(subset.loc[selected_web_ids])

    return full_corpus_updated

