import scipy.io.wavfile

def read_file(path, log):
    """
    Reads single wav file
    :param path: path to wav file
    :param log: logger
    :return: signal and sample rate
    """
    log.info("In read_file: path %s", path)
    sample_rate, signal = scipy.io.wavfile.read(path)
    return signal, sample_rate

def read_files(paths, log):
    """
    Reads all wav files in list
    :param paths: list of paths to wav files
    :param log: logger
    :return: list of tuples (signal, sample_rate) per file
    """
    log.info("In read_files: paths len %d", len(paths))
    data = [read_file(path, log) for path in paths]
    log.info("In read_files: successfully read all files")
    return data