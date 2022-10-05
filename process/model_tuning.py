"""_summary_
"""
import logging, os, sys, time
from utils.decorators import *
import pandas as pd
import numpy as np
from process.helpers import *
from pprint import pprint


logging.basicConfig(filename='./logs/temp_log.log', 
format=os.environ['logging_format'], 
level=os.environ['logging_level'],
filemode=os.environ['logging_filemode'])

logger = logging.getLogger(__name__)
handler=logging.StreamHandler(sys.stdout)
handler.setLevel(logging.os.environ['logging_level'])
formatter = logging.Formatter(os.environ['logging_format'])
logger.addHandler(handler)


@timing('TIMER - Iterating LDA Model over the range of topics, alpha, and beta parameter values: ')
def grid_search(input_dict: dict):

    download_path = os.path.join(os.getcwd(), 'model_outputs', 'ModelRun' + ' ' + os.environ['output_folder_timestamp'])
    csv_filepath = os.path.join(download_path, 'lda_tuning_results.csv')

    corpus = input_dict['corpus']
    id2word = input_dict['id2word']
    data_lemmatized = input_dict['data_lemmatized']

    grid = {}
    grid['Validation_Set'] = {}

    # Topics range
    min_topics = 2
    max_topics = 11
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)

    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    # alpha = []
    alpha.append('symmetric')
    alpha.append('asymmetric')

    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    # beta = []
    beta.append('symmetric')

    # Validation sets
    corpus_sets = [
                # gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)),
                corpus, 
                ]

    corpus_title = [
                    # '75% Corpus', 
                    '100% Corpus', 
                    ]

    model_results = {'Validation_Set': [],
                    'Topics': [],
                    'Alpha': [],
                    'Beta': [],
                    'Coherence': []
                    }

    total=(len(beta)*len(alpha)*len(topics_range)*len(corpus_title))
    run_no = 0

    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                  k=k, a=a, b=b, 
                                                  data_lemmatized=data_lemmatized)
                    
                    run_no += 1
                    logger.info(f'Grid Search: {run_no} finished out of {total}')  

                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)

    df_model_results = pd.DataFrame(model_results)
    
    df_model_results.to_csv(csv_filepath, index=False)

    to_return = input_dict

    to_return['model_results'] = df_model_results

    return to_return

