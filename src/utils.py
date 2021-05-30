import os
import numpy as np
from logging import getLogger, INFO, DEBUG, Formatter, FileHandler, StreamHandler

from config import CFG

def get_logger(name, logfile='log.txt'):
    """

    Create and return Logger module.

    Args:
        name (str): module name logger is envoked in
        logfile (str): output file name

    Returns:
        logger: Logger 
    """
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


def seed_everything(seed=42):
    """
    Fix seed values
    
    Args:
        seed (int): seed value
    
    Returns:
        None    
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed
    


if __name__ == '__main__':
    logger = get_logger(__name__)
    logger.debug('Logger was created successfully.')