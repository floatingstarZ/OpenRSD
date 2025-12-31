from M_Loggers.utils import AutoLogger, FlashLogger, logger_root, to_array
# CLv2_logger = AutoLogger(logger_root, 'CLv2_logger', mode='w',
#                          timestamp=False, cache_limit=100000)
class GloLogger():
    logger = AutoLogger(logger_root, 'None', mode='w', timestamp=False, cache_limit=100000)

# Logger 7
L1_E1_e1_check_UniRS_cropper = False
if L1_E1_e1_check_UniRS_cropper:
    prefix = 'L1_E1_e1_check_UniRS_cropper'
    GloLogger.logger = AutoLogger(
        logger_root,
        prefix,
        msg=True,
        cache_limit=20,
        timestamp=False)

# Logger 7
L2_check_UniRS_image = False
if L2_check_UniRS_image:
    prefix = 'L2_check_UniRS_image_GT_Epoch1'
    GloLogger.logger = AutoLogger(
        logger_root,
        prefix,
        msg=True,
        cache_limit=50,
        timestamp=False)

# Logger 8
L3_check_meta_rcnn = False
if L3_check_meta_rcnn:
    prefix = 'L3_check_meta_rcnn'
    GloLogger.logger = AutoLogger(
        logger_root,
        prefix,
        msg=True,
        cache_limit=50,
        timestamp=False)
