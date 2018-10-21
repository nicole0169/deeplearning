#!/usr/bin/env python
# -*- coding: UTF-8 -*-


class Training(object):
    def __init__(self):
        self.weight = 0.2

    def __str__(self):
        return 'weight\t:%s\n' % self.weight

    def predict(self, input_vec):
        return self.weight * input_vec

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weight(input_vec, output, label, rate)

    def _update_weight(self, input_vec, output, label, rate):
        self.weight += rate * ((label - output) / input_vec)
        # self.weight += rate * (label - output) * input_vec


def get_training_dataset():
    input_vecs = [1, 3, 4, 7]
    labels = [6.28, 18.84, 25.12, 43.96]
    # labels = [3.1416, 28.274, 50.265, 153.94]
    return input_vecs, labels


def train_and_percptron():
    input_vecs, labels = get_training_dataset()
    p = Training()
    p.train(input_vecs, labels, 50, 0.1)
    return p


if __name__ == '__main__':
    circle_train = train_and_percptron()
    print circle_train
    print 'r=2, C :%s' % circle_train.predict(2)
    print 'r=5, C :%s' % circle_train.predict(5)
