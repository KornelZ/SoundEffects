import numpy as np

class DTW(object):

    def __init__(self):
        self.vectors = []

    def _distance(self, a, b):
        """
        Calculates edit distance between two matrices (second norm of vector)
        :param a:
        :param b:
        :return:
        """
        dist_matrix = np.ndarray((a.shape[0] + 1, b.shape[0] + 1))
        dist_matrix.fill(np.finfo(np.float32).max)
        dist_matrix[0, 0] = 0

        for i in range(1, dist_matrix.shape[0]):
            for j in range(1, dist_matrix.shape[1]):
                dist_matrix[i, j] = np.linalg.norm(a[i - 1, :] - b[j - 1, :]) +\
                                    min([dist_matrix[i - 1, j - 1], dist_matrix[i - 1, j], dist_matrix[i, j - 1]])
        return dist_matrix[-1, -1]

    def train(self, train_data, train_labels):
        """
        Trains and constructs DTW vector list
        :param train_data: matrix of sequences
        :param train_labels: vector of target labels (not one-hot)
        :return:
        """
        _, indices, counts = np.unique(train_labels, return_index=True, return_counts=True)
        for i in range(len(indices)):
            index = indices[i]
            min_avg = np.finfo(np.float32).max
            to_add = 0
            for j in range(index, index + counts[i]):
                avg = 0
                for k in range(index, index + counts[i]):
                    if j != k:
                        avg += self._distance(train_data[j, :, :], train_data[k, :, :])
                avg /= counts[i]
                if avg < min_avg:
                    min_avg = avg
                    to_add = j
            print("Adding {} to dtw db".format(to_add))
            self.vectors.append(train_data[to_add, :])

    def test(self, test_data, test_labels):
        """
        Calculates accuracy of classification of test_data
        :param test_data: matrix of sequences
        :param test_labels: expected labels (not one-hot)
        :return: Accuracy % of classification
        """
        acc = 0
        for i in range(len(test_labels)):
            distances = []
            for j in range(len(self.vectors)):
                distances.append(self._distance(self.vectors[j], test_data[i, :, :]))
            r = np.argmin(distances)
            print("Result of {}: {}".format(test_labels[i], r))
            acc += int(test_labels[i] == r)
            print("Accuracy", acc / (i + 1))


        print("Accuracy", acc / len(test_labels))
        return acc / len(test_labels)

