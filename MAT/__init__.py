import logging
import sys
import torch as tr

# Init module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(logger_handler)
logger_handler.setFormatter(
    logging.Formatter('%(asctime)s [%(levelname)-5s] %(module)-10s %(funcName)-10s %(message)s'))

# init device
DEVICE = "cuda:0" if tr.cuda.is_available() else "cpu"
