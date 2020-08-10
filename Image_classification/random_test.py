import random as rand
import os
import numpy as np
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
'''
saves random numbers 0-500 for0 1 million iterations
'''
random_nums = []

for x in range(10000000):
    random_nums.append(rand.randint(0,500))

np.savetxt(__location__ + '\\random_distribution.txt',random_nums)