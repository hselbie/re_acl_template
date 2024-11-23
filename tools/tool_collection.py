import re
from utils import create_logger
from langchain.tools import tool
import uuid

logger = create_logger.init_logger(__name__, testing_mode='DEBUG')

@tool
def add(x: int, y: int) -> int:
    '''test function'''
    result = x+y
    return result