"""
This module contains functions and variables required to establish connections to a local
PostgreSQL databas, execute one or multiple SQL queries and save subsequent reports
to DataFrame objects.

"""

import pandas as pd
import psycopg2

# LOGIN DETAILS #
db = 'predicting_papers_v1'
host = 'localhost'
user = 'liza'
password = '4Eiadb9'


# DATABASE CONNECTION FUNCTION
def database_connect(db_name, host_name, user_name, user_password):
    """
    Connects to database via specific user profile
    Args:
        db_name: Database to which connection should be established
        host_name: Host name for database
        user_name: Username for login
        user_password: Password for user profile

    Returns:
        connection: Database connection

    """
    # Establishes database connection
    print('Connecting to database...')
    try:
        connection = psycopg2.connect(dbname=db_name,
                                      host=host_name,
                                      user=user_name,
                                      password=user_password,
                                      )
        print('\nConnection successful')

    except Exception as e:
        print('Connection unsuccessful - please check connection and try again')

    return connection


# DATABASE QUERY FUNCTION
# Used to execute one or more SQL scripts and save result to list of DataFrame objects
def db_to_df(script_list, connection):
    """
    Accesses database using pre-existing connection and executes provided list of SQL queries.
    Saves each resulting report to a Dataframe.

    Args:
        script_list (lst): List of SQL scripts to be executed
        connection: Database connection variable name
    Returns:
        all_reports: List of DataFrames retrieved by each SQL script
    """
    # Initialises empty dictionary for compilation of report DataFrames
    result = []

    # Iterates through each SQL script in list and assigns integer index value
    for script_num, script in enumerate(script_list):
        print(script_num,
              script
              )

        # Queries the database and saves retrieved report to a DataFrame
        report_df = pd.read_sql(script, connection)
        print(report_df.columns,
              report_df.iloc[0:5]
              )

        if len(script_list) == 1:
            result = report_df
        else:
            # Appends each DataFrame to list of DataFrames
            result.append(report_df)
    print('Download complete')
    connection.close()

    return result


# SQL SCRIPTS
create_db_tables = [
    '''
    CREATE TABLE final_pubmed_corpus
    (
      pmid        BIGINT,
      corpus_text TEXT,
      title       TEXT,
      abstract    TEXT,
      year        TEXT,
      authors     TEXT,
      journal     TEXT,
      volume      TEXT,
      issue       TEXT,
      pages       TEXT,
      doi         TEXT,
      pmc         TEXT,
      web_id      DOUBLE PRECISION
    );
    ''',
    '''
    CREATE TABLE final_web_corpus
    (
      web_id      BIGINT,
      url         TEXT,
      pmid        BIGINT,
      title       TEXT,
      content     TEXT,
      corpus_text TEXT
    );

    CREATE TABLE pmid_web_links_full
    (
      pmid      BIGINT,
        final_url TEXT
    );
    ''',
    '''
    CREATE TABLE pubmed_data_full
    (
      PMID     BIGINT,
      Year     TEXT,
      Title    TEXT,
      Abstract TEXT,
      Authors  TEXT,
      Journal  TEXT,
      Volume   TEXT,
      Issue    TEXT,
      Pages    TEXT,
      DOI      TEXT,
      PMC      DOUBLE PRECISION
    );
    ''',
    '''
    CREATE TABLE vaccine_pmids_original
    (
      "PMID" BIGINT
    );
    ''',
    '''
    CREATE TABLE web_data_full
    (
      processed_content TEXT,
      final_url         TEXT,
      pmid              BIGINT
    );
    ''',
]

original_dataset_scripts = ['SELECT * FROM vaccine_pmids_original',
                            'SELECT * FROM pubmed_data_full',
                            'SELECT * FROM web_data_full',
                            'SELECT * FROM pmid_web_links_full',
                            ]

final_corpora_scripts = ['SELECT * FROM final_pubmed_corpus',
                         'SELECT * FROM final_web_corpus',
                         'SELECT * FROM final_links',
                         ]

