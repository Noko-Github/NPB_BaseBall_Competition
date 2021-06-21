import utils
import config

if __name__ == '__main__':

	# load config
	cfg = config.CFG()

	# set seed
	seed_everything(cfg.seed)

	# load logger
	logger = utils.get_logger(__name__)