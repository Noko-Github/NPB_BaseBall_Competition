import os
import sys
from pathlib import Path

import pandas as pd

import utils
import config

if __name__ == '__main__':

	# add work directory
	workdir = Path(__file__).parent.as_posix()
	sys.path.append(workdir)

	# load config
	CFG = config.CFG()

	# set seed
	utils.seed_everything(CFG.SEED)

	# load logger
	logger = utils.get_logger(__name__)
 
	# data load
	train_df = pd.read_csv(os.path.join(CFG.INPUT_PATH, 'train_data.csv'), index_col=0)
	test_df = pd.read_csv(os.path.join(CFG.INPUT_PATH, 'test_data.csv'), index_col=0)
	game_df = pd.read_csv(os.path.join(CFG.INPUT_PATH, 'game_info.csv'), index_col=0)