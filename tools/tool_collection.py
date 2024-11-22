import re
from utils import create_logger
import uuid

logger = create_logger.init_logger(__name__, testing_mode='DEBUG')

def add(x: int, y: int) -> int:
    '''test function'''
    result = x+y
    return result