from file_io import wav_io
from preproc import features
from networks import rnn, network_util
from plots import confusion_matrix
import numpy as np
import os
import dtw
import sys

opts = {
    #general
    "path": "F:\\SpeechDatasets\\data\\close",
    "alg": "rnn",
    "trials": 1,
    "save_model": True,
    "save_path": "tmp/model.tf",
    #mfcc
    "frame_len": 0.025,
    "frame_interval": 0.01,
    "num_filters": 26,
    "num_mfcc": 13,
    "use_delta": False,
    "remove_mfcc": True,
    #neural net
    "dropout": 0.5,
    "batch_size": 8,
    "lr": 0.001,
    "l2": 0.01,
    "epochs": 1200,
    "train_size": 3,
    "valid_size": 2,
    #rnn
    "rnn_layers": [200, 120]
}

def get_mfcc(data):
    feat_extractor = features.FeatureExtractor(
        frame_len=opts["frame_len"],
        frame_interval=opts["frame_interval"],
        num_filters=opts["num_filters"],
        num_mfcc=opts["num_mfcc"],
        use_delta=opts["use_delta"],
        remove_first_mfcc_coeff=opts["remove_mfcc"],
        low_freq=0
    )
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
            if j < opts["train_size"]:
                s_t.append(feat_extractor.get_mfcc(signal, sample_rate))
                labels_train.append(i)
            elif j < opts["train_size"] + opts["valid_size"]:
                s.append(feat_extractor.get_mfcc(signal, sample_rate))
                labels_valid.append(i)
            j += 1
        speakers_train.append(s_t)
        speakers_valid.append(s)
        i += 1

    def get_max_seq_len(speakers):
        return (max(speakers, key=lambda x: x.shape[0])).shape[0]

    def pad(speaker, max_seq_len):
        t = max_seq_len - speaker.shape[0]
        return np.pad(speaker, pad_width=((0, t), (0, 0)), mode="constant")

    print(i)
    speakers_train = [y for x in speakers_train for y in x]
    speakers_valid = [y for x in speakers_valid for y in x]
    max_seq_len = get_max_seq_len(speakers_train + speakers_valid)
    speakers_train = list(map(lambda x: pad(x, max_seq_len), speakers_train))
    speakers_valid = list(map(lambda x: pad(x, max_seq_len), speakers_valid))
    print(len(speakers_train))
    speakers_train = np.stack(speakers_train, axis=0)
    speakers_valid = np.stack(speakers_valid, axis=0)
    num_classes = i
    return speakers_train, labels_train, speakers_valid, labels_valid, num_classes


def run_rnn(train, valid, num_inputs, num_classes):
    speakers_train, labels_train = train
    speakers_valid, labels_valid = valid
    net = rnn.Rnn(
        num_inputs=num_inputs,
        num_classes=num_classes,
        layers=opts["rnn_layers"],
        dropout=opts["dropout"],
        epochs=opts["epochs"],
        l2_coef=opts["l2"],
        learning_rate=opts["lr"],
        batch_size=opts["batch_size"],
        save_model=opts["batch_size"],
        save_path=opts["save_path"],
        max_seq_len=speakers_train.shape[1]
    )
    input = speakers_train

    labels_train = network_util.one_hot_encode(np.array(labels_train),
                                               num_classes=num_classes)
    labels_valid = network_util.one_hot_encode(np.array(labels_valid),
                                               num_classes=num_classes)
    results = []
    for i in range(opts["trials"]):
        perm = np.random.permutation(len(labels_train))
        labels_train = labels_train[perm]
        input = input[perm, :]
        net.build()
        acc, epoch, pred = net.train(input, labels_train, speakers_valid, labels_valid)
        results.append((acc, epoch))
        net.close()
    print(results)


def run_dtw(train, valid):
    dtw_alg = dtw.DTW()
    dtw_alg.train(train[0], train[1])
    dtw_alg.test(valid[0], valid[1])


def run():
    PATH = opts["path"]
    dirs = [os.path.join(PATH, p) for p in os.listdir(PATH)]
    dirs = [(p, os.listdir(p)) for p in dirs]
    dirs = [[os.path.join(path, filename) for filename in filenames]
            for path, filenames in dirs]
    print(dirs)
    data = [wav_io.read_files(d) for d in dirs]
    print(data)
    if opts["alg"] == "rnn" or opts["alg"] == "dtw":
        num_inputs = opts["num_mfcc"] if not opts["use_delta"] else 3 * opts["num_mfcc"]

        speakers_train, labels_train, speakers_valid, labels_valid, num_classes \
            = get_mfcc(data)
        if opts["alg"] == "rnn":
            run_rnn((speakers_train, labels_train),
                    (speakers_valid, labels_valid),
                    num_inputs, num_classes)
        else:
            run_dtw((speakers_train, labels_train),
                    (speakers_valid, labels_valid))
    else:
        raise ValueError("Invalid algorithm {}".format(opts["alg"]))


if __name__ == "__main__":
    args = sys.argv[1:]
    try:
        for i in range(0, len(args), 2):
            if opts[args[i]] is int:
                opts[args[i]] = int(args[i + 1])
            elif opts[args[i]] is bool:
                opts[args[i]] = bool(args[i + 1])
            elif opts[args[i]] is float:
                opts[args[i]] = float(args[i + 1])
            else:
                opts[args[i]] = args[i + 1]
    except IndexError:
        print("Usage: {} <arg1-key> <arg2-value>...".format(os.path.basename(__file__)))
        exit(1)
    print("*" * 20)
    print("Running with following arguments:")
    for key in opts:
        print("{}: {}".format(key, opts[key]))
    print("*" * 20)
    run()



