"""_summary_
"""
import logging, os, sys, time
from utils.decorators import *
import pandas as pd
from process.helpers import *
from wordcloud import WordCloud
import matplotlib.pyplot as plt 

logging.basicConfig(filename='./logs/temp_log.log', 
format=os.environ['logging_format'], 
level=os.environ['logging_level'],
filemode=os.environ['logging_filemode'])

logger = logging.getLogger(__name__)
handler=logging.StreamHandler(sys.stdout)
handler.setLevel(logging.os.environ['logging_level'])
formatter = logging.Formatter(os.environ['logging_format'])
logger.addHandler(handler)


@timing('TIMER - Creating Wordcloud from BBC Comments: ')
def create_wordcloud(input_list:list):

    download_path = os.path.join(os.getcwd(), 'model_outputs', 'ModelRun' + ' ' + os.environ['output_folder_timestamp'])
    image_filepath = os.path.join(download_path, 'wordcloud.png')
    # Join the different processed titles together.
    long_string = ','.join(input_list)
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    # wordcloud.to_image()
    plt.figure(figsize= (20,7))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.title("Common 100 words in comments", pad = 14, weight = 'bold')
    plt.savefig(image_filepath)
    logger.info(f"Saving wordcloud output image")
