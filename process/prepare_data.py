"""_summary_
"""
import logging, os, sys, time
from utils.decorators import *
import pandas as pd
from process.helpers import *
import gensim
from gensim.utils import simple_preprocess
# NLTK Stop words
import nltk
import spacy
import gensim.corpora as corpora

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
from nltk.corpus import stopwords

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
def construct_corpus_and_mappings(input_list: list):

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(input_list))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    stop_words = stopwords.words('english')
    stop_words.extend(['now', 'will'])

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    to_return = {
                'corpus': corpus, 
                'id2word': id2word, 
                'data_lemmatized': data_lemmatized, 
                 }

    return to_return

@timing('TIMER - Transforming BBC Comments DataFrame: ')
def transform_comments_dataframe(dataframe: pd.DataFrame):

    dataframe = dataframe.astype({'uprating': int, 'downrating': int})
    dataframe['total_rating'] = dataframe['uprating'] + dataframe['downrating']
    # Remove less-rated comments
    dataframe = dataframe[dataframe['total_rating']>=10]

    # Remove short reply comments
    dataframe = dataframe[dataframe['comment'].str.len()>150]

    # Remove unpublished comments
    dataframe = dataframe[ ~ dataframe['comment'].str.contains('This comment was removed because')]

    # sample only top 100 rated comments
    dataframe = dataframe.sort_values(['total_rating'], ascending=False)[:100]

    # Remove the columns
    dataframe = dataframe.drop(columns=[
                                'username', 'postdate', 'uprating', 
                                'downrating', 'reactid', 'reactid_len', 
                                'total_rating',
                                        ], 
                                axis=1).rename(columns={'comment': 'comment_text'})

    # Remove punctuation
    dataframe['comment_text_processed'] = dataframe['comment_text'].map(lambda x: re.sub('[,\.!?]', '', x))

    # Convert the titles to lowercase
    dataframe['comment_text_processed'] = dataframe['comment_text_processed'].map(lambda x: x.lower())

    list_of_comment_texts = dataframe['comment_text_processed'].values.tolist()

    return list_of_comment_texts