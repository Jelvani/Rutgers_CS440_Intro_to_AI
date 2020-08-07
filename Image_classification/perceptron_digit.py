import matplotlib.pyplot as plt
import read_data
import get_features
import numpy as np
import math
import random as rand
import os

class neuron(size = 28*28):
    weights = np.random.rand(size+1)
    def sum():
        return weights.