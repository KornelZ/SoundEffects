import speechpy
import numpy as np
import logging
from config import Config


class FeatureExtractor(object):

    def __init__(self, frame_len, frame_interval, num_mfcc, num_filters,
                 low_freq, remove_first_mfcc_coeff, use_delta):
        self.frame_len = frame_len
        self.frame_interval = frame_interval
        self.num_mfcc = num_mfcc
        self.num_filters = num_filters
        self.low_freq = low_freq
        self.remove_first_mfcc_coeff = remove_first_mfcc_coeff
        self.use_delta = use_delta

    def get_mfcc(self, wave, sample_rate):
        """
        Finds MFCC features of wave signal
        :param wave: audio signal read from .wav file
        :param sample_rate: sample rate of the signal
        :return: numpy arr of size (num of frames x num of mfcc) where
            num of frames and mfcc are set in config, if delta delta are used
            then num of mfcc is three times bigger
        """
        log = logging.getLogger(Config.LOG_NAME)
        log.info("In get_mfcc, wave shape %s, sample rate %s", wave.shape, sample_rate)
        mfcc = speechpy.feature.mfcc(
            signal=wave,
            sampling_frequency=sample_rate,
            frame_length=self.frame_len,
            frame_stride=self.frame_interval,
            num_cepstral=self.num_mfcc,
            num_filters=self.num_filters,
            low_frequency=self.low_freq,
            dc_elimination=self.remove_first_mfcc_coeff
        )
        log.info("In get_mfcc, mfcc shape %s", mfcc.shape)
        if self.use_delta:
            log.info("In get_mfcc, using delta-delta coeffs")
            mfcc = speechpy.feature.extract_derivative_feature(mfcc)
            log.info("In_get_mfcc, delta_mfcc shape %s", mfcc.shape)
            mfcc = np.reshape(mfcc, (mfcc.shape[0], mfcc.shape[1] * mfcc.shape[2]))
            log.info("In_get_mfcc, delta_mfcc after reshaping %s", mfcc.shape)
        return mfcc

    def get_spectrogram(self, normalize):
        """
        Based on A. Nagrani*, J. S. Chung*, A. Zisserman, VoxCeleb: a large-scale speaker identification dataset,
        INTERSPEECH, 2017
        :param normalize: normalizes each frequency bin by mean and std
        :return:
        """
        pass



