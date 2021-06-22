import os
import numpy as np
import random
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
    file_handler = FileHandler(os.path.join(CFG.LOG_DIR, logfile))
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
    np.random.seed(seed)


def reduce_mem_usage(df, verbose=False):
    """
    Reduce memory useage for dataframe
    
    Args:
        seed (int): seed value
    
    Returns:
        None    
    """
    start_mem = df.memory_usage().sum() / 1024**2
    cols = df.columns.to_list()
    df_1 = df.select_dtypes(exclude=['integer', 'float'])
    df_2 = df.select_dtypes(include=['integer']).apply(pd.to_numeric, downcast='integer')
    df_3 = df.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')
    df = df_1.join([df_2, df_3]).loc[:, cols]
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('{:.2f}Mb->{:.2f}Mb({:.1f}% reduction)'.format(
            start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
    


if __name__ == '__main__':
    logger = get_logger(__name__)
    logger.debug('Logger was created successfully.')