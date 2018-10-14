import speechpy
import numpy as np
import logging

def get_mfcc(config, wave, sample_rate):
    """
    Finds MFCC features of wave signal
    :param config: configuration class instance
    :param wave: audio signal read from .wav file
    :param sample_rate: sample rate of the signal
    :return: numpy arr of size (num of frames x num of mfcc) where
        num of frames and mfcc are set in config, if delta delta are used
        then num of mfcc is three times bigger
    """
    log = logging.getLogger(config.LOG_NAME)
    log.info("In get_mfcc, wave shape %s, sample rate %s", wave.shape, sample_rate)
    mfcc = speechpy.feature.mfcc(
        signal=wave,
        sampling_frequency=sample_rate,
        frame_length=config.FRAME_LENGTH,
        frame_stride=config.FRAME_INTERVAL,
        num_cepstral=config.NUM_OF_MFCC,
        num_filters=config.FILTERBANK_SIZE,
        low_frequency=config.LOW_FREQ,
        high_frequency=config.HIGH_FREQ,
        dc_elimination=config.REMOVE_FIRST_MFCC_COEFF
    )
    log.info("In get_mfcc, mfcc shape %s", mfcc.shape)
    if config.USE_DELTA_AND_DELTA_DELTA:
        log.info("In get_mfcc, using delta-delta coeffs")
        mfcc = speechpy.feature.extract_derivative_feature(mfcc)
        log.info("In_get_mfcc, delta_mfcc shape %s", mfcc.shape)
        mfcc = np.reshape(mfcc, (mfcc.shape[0], mfcc.shape[1] * mfcc.shape[2]))
        log.info("In_get_mfcc, delta_mfcc after reshaping %s", mfcc.shape)
    return mfcc



