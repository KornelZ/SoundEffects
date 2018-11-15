from file_io import wav_io
from preproc.features import FeatureExtractor

"""
def read_wav(path):
    return scipy.io.wavfile.read(path)


def write_wav(path, data, sample_rate):
    data = data.astype(np.int16) #type int16 signed wav file
    scipy.io.wavfile.write(path, sample_rate, data)


rate, wav = read_wav("KZ2.wav")
echo = effect.Echo(rate, 0.40, 0.40, 5, False)
write_wav("echoed.wav", echo.apply(wav), rate)

rate, wav = read_wav("KZ2.wav")
echo = effect.Echo(rate, delay=0.40, decay=0.40, repeats=2, prolong=False)
noise = effect.Noise(mu=0, sigma=25, effect=echo)
write_wav("echo_then_noise.wav", noise.apply(wav), rate)

rate, wav = read_wav("KZ2.wav")
noise = effect.Noise(mu=0, sigma=25)
echo = effect.Echo(rate, delay=0.40, decay=0.40, repeats=2, prolong=False, effect=noise)
write_wav("noise_then_echo.wav", echo.apply(wav), rate)
"""
TEST_FILE_PATH = "sample_wav.wav"
TEST_FILE_SAMPLES = 408226
TEST_FILE_RATE = 8000

def read_file_test():
    signal, sample_rate = wav_io.read_file(TEST_FILE_PATH)
    print(read_file_test.__name__)
    print("Signal shape", signal.shape)
    print("Sample rate", sample_rate)
    assert sample_rate == TEST_FILE_RATE
    assert signal.shape[0] == TEST_FILE_SAMPLES

read_file_test()

def mfcc_base_params_test():
    signal, sample_rate = wav_io.read_file(TEST_FILE_PATH)
    expected_features = 13
    feat_extractor = FeatureExtractor(
        frame_len=0.025,
        frame_interval=0.01,
        num_filters=26,
        num_mfcc=13,
        use_delta=False,
        remove_first_mfcc_coeff=False,
        low_freq=0
    )
    mfcc = feat_extractor.get_mfcc(wave=signal,
                                   sample_rate=sample_rate)
    print("Mfcc shape", mfcc.shape)
    expected_frames = len(signal) // (sample_rate * 0.01) - 2
    print("Num of frames", mfcc.shape[0])
    print("Num of features", mfcc.shape[1])
    assert mfcc.shape == (expected_frames, expected_features)

mfcc_base_params_test()

def mfcc_delta_params_test():
    signal, sample_rate = wav_io.read_file(TEST_FILE_PATH)
    expected_features = 39
    expected_frames = 5100
    feat_extractor = FeatureExtractor(
        frame_len=0.025,
        frame_interval=0.01,
        num_filters=26,
        num_mfcc=13,
        use_delta=True,
        remove_first_mfcc_coeff=False,
        low_freq=0
    )
    mfcc = feat_extractor.get_mfcc(wave=signal,
                                   sample_rate=sample_rate)
    print("Mfcc shape", mfcc.shape)
    print("Num of frames", mfcc.shape[0])
    print("Num of features", mfcc.shape[1])
    assert mfcc.shape == (expected_frames, expected_features)

mfcc_delta_params_test()