from file_io import wav_io
from preproc import features
from networks import rnn, network_util
import numpy as np
import os

PATH = "F:\\SpeechDatasets\\data\\close"
dirs = [os.path.join(PATH, p) for p in os.listdir(PATH)]
dirs = [(p, os.listdir(p)) for p in dirs]
dirs = [[os.path.join(path, filename) for filename in filenames]
        for path, filenames in dirs]
print(dirs)
data = [wav_io.read_files(d) for d in dirs]
print(data)

feat_extractor = features.FeatureExtractor(
    frame_len=0.025,
    frame_interval=0.01,
    num_filters=26,
    num_mfcc=13,
    use_delta=False,
    remove_first_mfcc_coeff=True,
    low_freq=0
)
speakers = []
n = 100
max = 0

def pad(a, n):
    global max
    t = n - a.shape[0]
    if max < a.shape[0]:
        max = a.shape[0]
    return np.pad(a, pad_width=((0, t), (0, 0)), mode="constant")

i = 0
labels = []
labels_t = []
speakers_t = []
for speaker in data:
    s = []
    s_t = []
    j = 0
    for signal, sample_rate in speaker:
        j += 1
        if j > 4:
            x = pad(feat_extractor.get_mfcc(signal, sample_rate), n)
            s_t.append(pad(feat_extractor.get_mfcc(signal, sample_rate), n))
            labels_t.append(i)
        else:
            s.append(pad(feat_extractor.get_mfcc(signal, sample_rate), n))
            labels.append(i)

    speakers.append(s)
    speakers_t.append(s_t)
    i += 1

print(i)
speakers = [y for x in speakers for y in x]
speakers_t = [y for x in speakers_t for y in x]
print(len(speakers))
speaker_arr = np.stack(speakers, axis=0)
speaker_test_arr = np.stack(speakers_t, axis=0)

net = rnn.Rnn(
        num_inputs=13,
        num_classes=i,
        layers=[256, 128],
        dropout=0.6,
        epochs=400,
        l2_coef=0.01,
        learning_rate=0.001,
        batch_size=8,
        save_model=True,
        save_path="tmp/model.tf",
        max_seq_len=n
    )
input = speaker_arr

labels = network_util.one_hot_encode(np.array(labels),
                                     num_classes=i)
labels_t = network_util.one_hot_encode(np.array(labels_t), num_classes=i)
perm = np.random.permutation(len(labels))
labels = labels[perm]
input = input[perm, :]
net.build()
net.train(input, labels, speaker_test_arr, labels_t)

#net.test(input[size:], labels[size:], model_path="tmp/model.tf-1000.meta")



