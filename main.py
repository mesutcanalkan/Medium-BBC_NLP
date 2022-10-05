import logging, os, sys
import pandas as pd
from configuration import configure_application


def run_tasks(news_url:str):

    configure_application.load_config_set_senv_variables()
    configure_application.create_application_folders()

    logging.basicConfig(filename='./logs/temp_log.log', 
    format=os.environ['logging_format'], 
    level=os.environ['logging_level'],
    filemode=os.environ['logging_filemode'])

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.os.environ['logging_level'])
    formatter = logging.Formatter(os.environ['logging_format'])
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    # Check if executable or running from terminal and set root project directory accordingly
    if getattr(sys, 'frozen', False):
        os.environ['root_dir'] = os.path.dirname(sys.executable)
    elif __file__:
        os.environ['root_dir'] = os.path.dirname(os.path.abspath(__file__))


    from process import (helpers, scraper, wordcloudify, prepare_data, model_base, model_tuning, model_final)

    logger.info(f"1-Scraping Comments")
    scraping_response = scraper.scrape_bbc_comments(news_url=news_url)
    list_of_comment_texts = prepare_data.transform_comments_dataframe(dataframe=scraping_response['data'])

    logger.info(f"2-Generating Wordcloud")
    wordcloudify.create_wordcloud(input_list=list_of_comment_texts)

    logger.info(f"3-Constructing Corpus")
    data_dict = prepare_data.construct_corpus_and_mappings(input_list=list_of_comment_texts)

    logger.info(f"4-Running Base Model")
    base_lda_model, base_coherence_model_lda = model_base.build_lda_model(input_dict=data_dict)

    logger.info(f"5-Running Grid Search")
    data_dict = model_tuning.grid_search(input_dict=data_dict)

    logger.info(f"6-Running Best Model")
    best_lda_model, best_coherence_model_lda = model_final.run_best_params(input_dict=data_dict)

    
if __name__ == "__main__":
    
    bbc_news_post = 'https://www.bbc.co.uk/news/business-63030208'
    
    run_tasks(news_url=bbc_news_post)