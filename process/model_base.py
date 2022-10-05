"""_summary_
"""
import logging, os, sys, time
from utils.decorators import *
import pandas as pd
from process.helpers import *
import gensim
from gensim.models import CoherenceModel
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

@timing('TIMER - Constructing Corpus and id2word from Top BBC Comments: ')
def build_lda_model(input_dict: dict):

    corpus = input_dict['corpus']
    id2word = input_dict['id2word']
    data_lemmatized = input_dict['data_lemmatized']

    logger.info('Building Base LDA Model')  
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=10, 
                                        random_state=100,
                                        chunksize=100,
                                        passes=10,
                                        per_word_topics=True)


    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())


    doc_lda = lda_model[corpus]
    Perplexity = lda_model.log_perplexity(corpus)
    # Compute Perplexity
    logger.info(f'Base LDA Model Perplexity: {Perplexity}')  
    # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    logger.info(f'Base LDA Model Coherence Score: {coherence_lda}')

    df_topic_composition = pd.DataFrame(lda_model.print_topics(), columns=['Topic', 'Composition'])
    download_path = os.path.join(os.getcwd(), 'model_outputs', 'ModelRun' + ' ' + os.environ['output_folder_timestamp'])
    csv_filepath = os.path.join(download_path, f'Base Model Composition Perplexity {Perplexity:.4f} Coherence {coherence_lda:.4f}.csv')
    df_topic_composition.to_csv(csv_filepath, index=False)

    return lda_model, coherence_model_lda


