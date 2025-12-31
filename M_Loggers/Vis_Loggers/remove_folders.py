import matplotlib.pyplot as plt

from MY_LOGGERS.utils import logger_root, AutoLogger
from commonlibs.common_tools import *
import numpy as np
import cv2

# CLv2_IOU_H07_L01
# CLv2_Base
# CLv2_IOU_H07_L01_Encoder
logger_name = 'CLv2_IOU_H07_L01_Encoder'
from MY_LOGGERS.utils import merge
logger_root = logger_root + '/' + logger_name
logger_dir = logger_root + '/' + logger_name

