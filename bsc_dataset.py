from file_io import wav_io
from preproc import features
from networks import rnn, network_util
from plots import confusion_matrix
import numpy as np
import os
import dtw

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

n = 100
max = 0

def pad(a, n):
    global max
    t = n - a.shape[0]
    if max < a.shape[0]:
        max = a.shape[0]
    return np.pad(a, pad_width=((0, t), (0, 0)), mode="constant")

i = 0
speakers_train = []
labels_train = []
labels_valid = []
speakers_valid = []
for speaker in data:
    s = []
    s_t = []
    j = 0
    for signal, sample_rate in speaker:
        if j < 3:
            x = pad(feat_extractor.get_mfcc(signal, sample_rate), n)
            s_t.append(pad(feat_extractor.get_mfcc(signal, sample_rate), n))
            labels_train.append(i)
        elif j < 5:
            s.append(pad(feat_extractor.get_mfcc(signal, sample_rate), n))
            labels_valid.append(i)
        j += 1

    speakers_train.append(s_t)
    speakers_valid.append(s)
    i += 1


print(i)
speakers_train = [y for x in speakers_train for y in x]
speakers_valid = [y for x in speakers_valid for y in x]
print(len(speakers_train))
speakers_train = np.stack(speakers_train, axis=0)
speakers_valid = np.stack(speakers_valid, axis=0)

#dtw_alg = dtw.DTW()
#dtw_alg.train(speakers_train, labels_train)
#dtw_alg.test(speakers_valid, labels_valid)

net = rnn.Rnn(
        num_inputs=13,
        num_classes=i,
        layers=[200, 120],
        dropout=0.5,
        epochs=1200,
        l2_coef=0.01,
        learning_rate=0.001,
        batch_size=8,
        save_model=True,
        save_path="tmp/model.tf",
        max_seq_len=n
    )
input = speakers_train

labels_train = network_util.one_hot_encode(np.array(labels_train),
                                     num_classes=i)
labels_valid = network_util.one_hot_encode(np.array(labels_valid), num_classes=i)
results = []
for i in range(10):
    perm = np.random.permutation(len(labels_train))
    labels_train = labels_train[perm]
    input = input[perm, :]
    net.build()
    acc, epoch, pred = net.train(input, labels_train, speakers_valid, labels_valid)
    results.append((acc, epoch))
#confusion_matrix(pred, labels_t, i)
    net.close()
print(results)
#net.test(input[size:], labels[size:], model_path="tmp/model.tf-1000.meta")"""



