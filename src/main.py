import os
import sys
from pathlib import Path

import pandas as pd

import utils
import preprocess
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
	train_df = pd.read_csv(os.path.join(CFG.INPUT_PATH, 'train_data.csv'))
	test_df = pd.read_csv(os.path.join(CFG.INPUT_PATH, 'test_data.csv'))
	game_df = pd.read_csv(os.path.join(CFG.INPUT_PATH, 'game_info.csv'), index_col=0)

	# reduce memory usage of dataframe
	train_df = utils.reduce_mem_usage(train_df)
	test_df = utils.reduce_mem_usage(test_df)
	game_df = utils.reduce_mem_usage(game_df)

	# delete duplicate rows
	train_df = train_df[~train_df.drop('id', axis=1).duplicated()]
	test_df = test_df[~test_df.drop('id', axis=1).duplicated()]

	# merge game_df
	train_df = pd.merge(train_df, game_df.drop(['bgTop', 'bgBottom'], axis=1), on='gameID', how='left')
	test_df = pd.merge(test_df, game_df.drop(['bgTop', 'bgBottom'], axis=1), on='gameID', how='left')

	# add team postfix to avoid same batter/picther name
	train_df = preprocess.add_team_postfix(train_df)
	test_df = preprocess.add_team_postfix(test_df)

	# trainとtestに共通のピッチャーを取\
	train_pitcher = set(train_df['pitcher'].unique())
	test_pitcher = set(test_df['pitcher'].unique())
	train_batter = set(train_df['batter'].unique())
	test_batter = set(test_df['batter'].unique())

	# merge train and test
	input_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
	del train_df, test_df, game_df

	input_df = preprocess.fill_na(input_df)

	base_df = preprocess.get_base_features(input_df, train_pitcher, test_pitcher, train_batter, test_batter)
	print(base_df)


	
	

