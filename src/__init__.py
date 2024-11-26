import logging

from .utils import *
from .loaders import *
from .node import *
from .tree import *
from .plot import *

def set_logger(path, args):
    logger = logging.getLogger(__name__)
    logging_format = logging.Formatter(
        fmt='[%(levelname)s] (%(asctime)s) %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p'
    )
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(path)
    
    stream_handler.setFormatter(logging_format)
    file_handler.setFormatter(logging_format)
    
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level=logging.INFO)
    
    logger.info('[WELCOME] Initialisation...')
    welcome_message = """VTree Learning, DFL."""
    logger.info(welcome_message)
