import re
from utils import create_logger
import uuid

logger = create_logger.init_logger(__name__, testing_mode='DEBUG')

import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

def add(x: int, y: int) -> int:
    '''test function'''
    result = x+y
    return result