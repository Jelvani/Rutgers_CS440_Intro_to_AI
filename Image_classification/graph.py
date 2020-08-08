import matplotlib.pyplot as plt
import os
import numpy as np

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def perceptron_digits():
    data = np.loadtxt(__location__ + '\\perceptron_digits_training_results.txt')
    mean = []
    stddev = []
    for x in range(0,len(data[0]),5):
        sum = 0
        std = []
        for y in range(5):
            std.append(data[0][x+y])
            sum += data[0][x+y]
        mean.append(sum/5)
        stddev.append(np.std(std))

    x = np.arange(0.1,1.1,0.1)
    h1 = plt.plot(x,mean)
    #h2 = plt.plot(x,stddev)
    #plt.ylim(0.6,0.8)

    plt.xticks(np.arange(0.1, 1.1, step=0.1))

    plt.title('Perceptron Training Accuracy')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Accuracy')
    plt.show(h1)

def perceptron_faces():
    data = np.loadtxt(__location__ + '\\perceptron_faces_training_results.txt')
    mean = []
    stddev = []
    for x in range(0,len(data[0]),5):
        sum = 0
        std = []
        for y in range(5):
            std.append(data[0][x+y])
            sum += data[0][x+y]
        mean.append(sum/5)
        stddev.append(np.std(std))

    x = np.arange(0.1,1.1,0.1)
    #h1 = plt.plot(x,mean)
    h2 = plt.plot(x,stddev)
    #plt.ylim(0.6,0.8)

    plt.xticks(np.arange(0.1, 1.1, step=0.1))

    plt.title('Perceptron Training Accuracy')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Accuracy')
    plt.show(h2)

def bayes_digits():
    data = np.loadtxt(__location__ + '\\bayes_digits_training_results.txt')
    mean = []
    stddev = []
    for x in range(0,len(data[0]),5):
        sum = 0
        std = []
        for y in range(5):
            std.append(data[0][x+y])
            sum += data[0][x+y]
        mean.append(sum/5)
        stddev.append(np.std(std))

    x = np.arange(0.1,1.1,0.1)
    h1 = plt.plot(x,mean)
    #h2 = plt.plot(x,stddev)
    #plt.ylim(0.6,0.8)

    plt.xticks(np.arange(0.1, 1.1, step=0.1))

    plt.title('Bayes Training Accuracy')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Accuracy')
    plt.show(h1)
perceptron_faces()