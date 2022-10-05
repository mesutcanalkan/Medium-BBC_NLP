"""_summary_
"""
import logging, os, sys, re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import time
from retrying import retry
from typing import Any, Callable
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import (ElementNotInteractableException,
                                        NoSuchElementException,
                                        StaleElementReferenceException,
                                        TimeoutException)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import gensim
from gensim.models import CoherenceModel


logging.basicConfig(filename='./logs/temp_log.log', 
format=os.environ['logging_format'], 
level=os.environ['logging_level'],
filemode=os.environ['logging_filemode'])

logger = logging.getLogger(__name__)
handler=logging.StreamHandler(sys.stdout)
handler.setLevel(logging.os.environ['logging_level'])
formatter = logging.Formatter(os.environ['logging_format'])
logger.addHandler(handler)

def string_to_boolean(input_string:str):

    return input_string=='True'

def make_chrome_instance(headless, download_path):
    
    # Configure browser
    options = Options()
    if os.environ['headless'].lower() == 'true':
        options.add_argument('--headless')
    options.add_argument(os.environ['window_size'])
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    # options.add_experimental_option('useAutomationExtension', False)
    # options.add_experimental_option('excludeSwitches', ['enable-automation'])
    # options.add_argument("--lang=de-DE")
    options.add_experimental_option("prefs", {
        "download.default_directory": download_path,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    })
    
    # Instatiate browser
    # chrome_v = chrome_version()
    # chrome_driver = driver_path + f"chromedriver{chrome_v}.exe"
    # service = Service(executable_path=chrome_driver)
    # service = Service(executable_path=driver_path + 'chromedriver105.exe')
    # driver = webdriver.Chrome(service=service, options=options)
    driver = webdriver.Chrome(ChromeDriverManager().install())
    
    # Allow for downloads in headless mode
    driver.command_executor._commands["send_command"] = (
        "POST", '/session/$sessionId/chromium/send_command')
    params = {
        'cmd':'Page.setDownloadBehavior', 
        'params': {
            'behavior': 'allow', 
            'downloadPath': download_path,
        },
    }
    driver.execute("send_command", params)

    return driver

def wait(attempts, delay):
    logger.warning('Attempt #%d failed, retrying...' % (attempts))
    return delay


@retry(stop_max_attempt_number=3, wait_fixed=500, wait_func=wait)
def wait_for_expected_element(
        driver: Any, 
        timeout: int, 
        by_method: str, 
        value: str, 
        EC_method: Callable=EC.presence_of_element_located, 
        fill_text: str=None,
        send_return_key=False,
        click: bool=False,
        return_element: bool=False, 
        return_element_text: bool=False,        
    ):
    try:
        # Find element
        myElem = WebDriverWait(driver, timeout).until(EC_method((by_method, value)))
        
        # Interact with element
        if fill_text:
            myElem.send_keys(Keys.CONTROL + "a")
            myElem.send_keys(Keys.DELETE) 
            myElem.send_keys(fill_text)
        if send_return_key:
            myElem.send_keys(Keys.RETURN)
        if click:
            myElem.click()
        if return_element:
            to_return = myElem
        elif return_element_text:
            to_return = myElem.text
        else:
            to_return = None
    
    except TimeoutException:
        message = f'A TimeoutException occured waiting for the following web element to load: {by_method} ({value}).'
        logger.error(message)
        raise TimeoutException(message)
    
    except NoSuchElementException:
        message = f'NoSuchElementException occured when trying to locate the following web element: {by_method} ({value}).'
        logger.error(message)
        raise NoSuchElementException(message)
    
    except StaleElementReferenceException:
        message = f'StaleElementException occured when trying to locate the following web element: {by_method} ({value}).'
        logger.error(message)
        raise StaleElementReferenceException(message)
    
    except ElementNotInteractableException:
        message = f'ElementNotInteractableException occured when trying to interact with the following web element: {by_method} ({value}).'
        logger.error(message)
        raise ElementNotInteractableException(message)

    else: 
        return to_return


def compute_coherence_values(corpus, dictionary, k, a, b, data_lemmatized):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=dictionary, coherence='c_v')
    
    return coherence_model_lda.get_coherence()

