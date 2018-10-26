
class Config(object):
    """Features"""
    FRAME_LENGTH = 0.025
    FRAME_INTERVAL = 0.01
    NUM_OF_MFCC = 13
    FILTERBANK_SIZE = 26
    LOW_FREQ = 0
    REMOVE_FIRST_MFCC_COEFF = False
    USE_DELTA_AND_DELTA_DELTA = True

    """Log"""
    LOG_NAME = "log.txt"
