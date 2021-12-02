import math
import numpy
import random
import torch
import torch.utils.data

from typing import Tuple

class CircleDataSet(torch.utils.data.Dataset):
    def __init__(self, input, output):
        self.i = input
        self.o = numpy.asarray(output)
        return

    def __len__(self):
        return len(self.i)

    def __getitem__(self, index) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return self.i[index,:], self.o[index,:]

class CircleFullDataSet(object):
    def __init__(self):
        samples_per_rotation = 8
        rotation_count = 100
        self.x = numpy.arange(0, (math.pi * 2) * rotation_count, math.pi*2/samples_per_rotation)
        self.x = self.x.reshape(self.x.shape[0], 1)
        self.y = numpy.zeros((self.x.shape[0], 2))
        for bi in range(self.x.shape[0]):
            self.y[bi, 0] = numpy.cos(self.x[bi]) * 0.5 + (random.random() - 0.5) * 0.001
            self.y[bi, 1] = numpy.sin(self.x[bi]) * 0.5 + (random.random() - 0.5) * 0.001

        self.train = CircleDataSet(self.x, self.y)
        self.valid = CircleDataSet(self.x, self.y)
        self.test = CircleDataSet(self.x, self.y)
        return

    @property
    def input_shape(self) -> Tuple[int]:
        return self.x.shape

    @property
    def output_shape(self) -> Tuple[int, int]:
        return self.y.shape

    @property
    def training_set(self) -> CircleDataSet:
        return self.train

    @property
    def validation_set(self) -> CircleDataSet:
        return self.valid
