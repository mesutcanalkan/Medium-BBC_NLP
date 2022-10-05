
import logging
import os, sys
import time

logging.basicConfig(filename='./logs/temp_log.log', format=os.environ['logging_format'], level=os.environ['logging_level'])
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.os.environ['logging_level'])
formatter = logging.Formatter(os.environ['logging_format'])
handler.setFormatter(formatter)
logger.addHandler(handler)
    
# Define decorator
def timing(message):
    def decorator(f):
        def wrap(*args, **kw):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            log_time(message + str(round(te-ts, 2)))
            return result
        return wrap
    return decorator


def log_time(s):
        logger.info(s)
        with open("timings.txt", "a") as file_object:
            # Append 'hello' at the end of file
            file_object.write('\n'+s)