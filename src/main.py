import utils
import config

if __name__ == '__main__':

	# load config
	cfg = config.CFG()

	# set seed
	seed_everything(cfg.seed)