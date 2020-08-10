import matplotlib.pyplot as plt
import os
import numpy as np

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def grapher(fname):
    data = np.loadtxt(__location__ + '\\'+fname)
    mean = []
    stddev = []
    time = []
    for x in range(0,len(data[0]),5):
        sum = 0
        t = 0
        std = []
        for y in range(5):
            std.append(data[0][x+y])
            sum += data[0][x+y]
            t+= data[1][x+y]
        mean.append(sum/5)
        time.append(t/5)
        stddev.append(np.std(std))

    x = np.arange(0.1,1.1,0.1)
    plt.plot(x,mean,linewidth=3)
    plt.title('Naive Bayes Faces Training Accuracy')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Accuracy')

    plt.show()
    plt.plot(x,stddev,linewidth=3)
    plt.title('Naive Bayes Faces Accuracy Standard Deviation')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Standard Deviation')

    plt.show()
    plt.plot(x,time,linewidth=3)
    plt.title('Naive Bayes Faces Training Time')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Time (seconds)')

    plt.show()

def rand_nums(fname):
    n_bins = 500
    data = np.loadtxt(__location__ + '\\'+fname)
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(data, bins=n_bins)
    plt.title('Python Random Module Distribution')
    plt.xlabel('Random Number Picked')
    plt.ylabel('Frequency')
    plt.show()

#grapher('bayes_faces_training_results.txt')
rand_nums('random_distribution.txt')