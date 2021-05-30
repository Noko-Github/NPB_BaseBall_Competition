import os
from logging import getLogger, INFO, DEBUG, Formatter, FileHandler, StreamHandler

from config import CFG

def get_logger(name, logfile='log.txt'):
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    
    # create file handler
    file_handler = FileHandler(os.path.join(CFG.log_dir, logfile))
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(
        Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s', '%Y-%m-%d %H:%M:%S'))

    # create stream handler
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(
        Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s', '%Y-%m-%d %H:%M:%S'))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


if __name__ == '__main__':
    logger = get_logger(__name__)
    logger.debug('Logger was created successfully.')