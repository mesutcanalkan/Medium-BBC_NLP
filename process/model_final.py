"""_summary_
"""
import logging, os, sys, time
from utils.decorators import *
import pandas as pd
import numpy as np
from process.helpers import *
import gensim
from gensim.models import CoherenceModel
from pprint import pprint
import pyLDAvis.gensim_models
import pickle 
import pyLDAvis


logging.basicConfig(filename='./logs/temp_log.log', 
format=os.environ['logging_format'], 
level=os.environ['logging_level'],
filemode=os.environ['logging_filemode'])

logger = logging.getLogger(__name__)
handler=logging.StreamHandler(sys.stdout)
handler.setLevel(logging.os.environ['logging_level'])
formatter = logging.Formatter(os.environ['logging_format'])
logger.addHandler(handler)


@timing('TIMER - Running LDA Model based on the best parameter values and saving outputs: ')
def run_best_params(input_dict: dict):

    corpus = input_dict['corpus']
    id2word = input_dict['id2word']
    data_lemmatized = input_dict['data_lemmatized']
    model_results = input_dict['model_results']

    model_results_max = model_results[model_results['Coherence'] == model_results['Coherence'].max()].reset_index(drop=True)

    num_topics = model_results_max.loc[0, 'Topics']

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics, 
                                            random_state=100,
                                            chunksize=100,
                                            passes=10,
                                            alpha=model_results_max.loc[0, 'Alpha'],
                                            eta=model_results_max.loc[0, 'Beta']) 

    # Print the Keyword in the num_topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    Perplexity = lda_model.log_perplexity(corpus)
    logger.info(lda_model.print_topics())
    # Compute Perplexity
    logger.info(f'Highest Performing LDA Model Perplexity: {Perplexity}')  
    # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    logger.info(f'Highest Performing LDA Model Coherence Score: {coherence_lda}')

    df_topic_composition = pd.DataFrame(lda_model.print_topics(), columns=['Topic', 'Composition'])
    download_path = os.path.join(os.getcwd(), 'model_outputs', 'ModelRun' + ' ' + os.environ['output_folder_timestamp'])
    csv_filepath = os.path.join(download_path, f'Best Model Composition Perplexity {Perplexity:.4f} Coherence {coherence_lda:.4f}.csv')
    df_topic_composition.to_csv(csv_filepath, index=False)

    LDAvis_data_filepath = os.path.join(download_path, 'ldavis_tuned_'+str(num_topics))

    logger.info(f'Saving HTML of the Highest Performing LDA Model: {coherence_lda}')

    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)

    pyLDAvis.save_html(LDAvis_prepared, LDAvis_data_filepath +'.html')

    return lda_model, coherence_model_lda

