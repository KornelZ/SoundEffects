import scipy.io.wavfile
import logging
from config import Config

def read_file(path):
    """
    Reads single wav file
    :param path: path to wav file
    :param log: logger
    :return: signal and sample rate
    """
    log = logging.getLogger(Config.LOG_NAME)
    log.info("In read_file: path %s", path)
    sample_rate, signal = scipy.io.wavfile.read(path)
    if len(signal.shape) != 1:
        raise ValueError("Signal shape cannot be > 1")
    return signal, sample_rate

def read_files(paths):
    """
    Reads all wav files in list
    :param paths: list of paths to wav files
    :param log: logger
    :return: list of tuples (signal, sample_rate) per file
    """
    log = logging.getLogger(Config.LOG_NAME)
    log.info("In read_files: paths len %d", len(paths))
    data = [read_file(path) for path in paths]
    log.info("In read_files: successfully read all files")
    return data