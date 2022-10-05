"""_summary_
"""
import logging, os, sys, time
from utils.decorators import *
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchWindowException

from process.helpers import *
from xpaths import bbc_xpaths


logging.basicConfig(filename='./logs/temp_log.log', 
format=os.environ['logging_format'], 
level=os.environ['logging_level'],
filemode=os.environ['logging_filemode'])

logger = logging.getLogger(__name__)
handler=logging.StreamHandler(sys.stdout)
handler.setLevel(logging.os.environ['logging_level'])
formatter = logging.Formatter(os.environ['logging_format'])
logger.addHandler(handler)


@timing('TIMER - Scraping BBC Comments: ')
def scrape_bbc_comments(news_url:str):

    download_path = os.path.join(os.getcwd(), 'model_outputs', 'ModelRun' + ' ' + os.environ['output_folder_timestamp'])

    driver = make_chrome_instance(headless=os.environ['headless'], 
                                    download_path=download_path)

    logger.info(f"Opening BBC News article: {news_url}")

    driver.get(news_url)

    try:

        wait_for_expected_element(
            driver=driver, 
            timeout=int(os.environ['timeout']), 
            value=bbc_xpaths.ACCEPT_COOKIES_BUTTON,
            by_method=By.XPATH,
            EC_method=EC.element_to_be_clickable,
            fill_text=None,
            click=True,
            return_element=False,
            )

    except:

        pass

    news_title = driver.title

    news_title = news_title.replace(' - BBC News', '') if news_title.endswith(' - BBC News') else news_title

    logger.info(f"News Title: {news_title}")

    logger.info(f"Viewing Comments")

    try: 
        wait_for_expected_element(
            driver=driver, 
            timeout=int(os.environ['timeout']), 
            value=bbc_xpaths.VIEW_COMMENTS_BUTTON,
            by_method=By.XPATH,
            EC_method=EC.element_to_be_clickable,
            fill_text=None,
            click=True,
            return_element=False,
            )
    except:
        pass

    iframes = driver.find_elements(By.XPATH, "//iframe")

    if iframes:
        WebDriverWait(driver, int(os.environ['timeout'])).until(EC.frame_to_be_available_and_switch_to_it(len(iframes)-1))

    total_comments = ''
    
    while total_comments=='':

        try:

            total_comments = wait_for_expected_element(
                                driver=driver, 
                                timeout=int(os.environ['timeout']), 
                                value=bbc_xpaths.TOTAL_COMMENTS,
                                by_method=By.XPATH,
                                EC_method=EC.presence_of_element_located,
                                fill_text=None,
                                click=False,
                                return_element=True,
                                ).text.strip()

        except TimeoutException:

            break


    logger.info(f"There are {total_comments} to collect!")

    logger.info(f"Iteratively clicking on more comments button until the page doesn't allow")

    while True:

        try: 

            wait_for_expected_element(
                driver=driver, 
                timeout=int(os.environ['timeout']), 
                value=bbc_xpaths.MORE_COMMENTS_BUTTON,
                by_method=By.XPATH,
                EC_method=EC.presence_of_element_located,
                fill_text=None,
                click=True,
                return_element=False,
                )

        except TimeoutException:

            break

    logger.info(f"Iteratively clicking on more replies button until the page doesn't allow")

    while True:

        try: 

            wait_for_expected_element(
                driver=driver, 
                timeout=int(os.environ['timeout']), 
                value=bbc_xpaths.MORE_REPLIES_BUTTON,
                by_method=By.XPATH,
                EC_method=EC.element_to_be_clickable,
                fill_text=None,
                click=True,
                return_element=False,
                )

        except TimeoutException:

            break

    logger.info(f"Putting all the comments info into a data frame")

    usernames = [(x.text) for x in driver.find_elements(By.XPATH, bbc_xpaths.COMMENT_USERNAME)]
    postdates = [(x.text) for x in driver.find_elements(By.XPATH, bbc_xpaths.COMMENT_DATE)]
    comments = [(x.text) for x in driver.find_elements(By.XPATH, bbc_xpaths.COMMENT_TEXT)]
    reactids = [(x.get_attribute('data-reactid')) for x in driver.find_elements(By.XPATH, bbc_xpaths.COMMENT_BODY)]
    upratings = [(x.text) for x in driver.find_elements(By.XPATH, bbc_xpaths.COMMENT_UPRATINGS)]
    downratings = [(x.text) for x in driver.find_elements(By.XPATH, bbc_xpaths.COMMENT_DOWNRATINGS)]

    df_comments = pd.DataFrame(zip(usernames, postdates, comments, upratings, downratings, reactids), 
    columns=['username', 'postdate', 'comment', 'uprating', 'downrating', 'reactid'])

    df_comments['reactid_len'] = df_comments['reactid'].str.len()

    driver.quit()

    comments_filepath = os.path.join(download_path, 'bbc_comments.xlsx')

    df_comments.to_excel(comments_filepath, index=False)

    to_return = {
                'download_path': download_path,
                'filepath': comments_filepath,
                'url': news_url,
                'title': news_title,
                'total_comments': total_comments,
                'data': df_comments,
                }

    return to_return