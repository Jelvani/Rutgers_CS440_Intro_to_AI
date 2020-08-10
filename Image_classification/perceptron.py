import matplotlib.pyplot as plt
import read_data
import get_features
import numpy as np
import math
import random as rand
import os
import time

'''
class below is for each class label (digits 0-9)

'''
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class neuron():
    def __init__(self,size = 28*28):
        self.size = size
        self.weights = np.zeros(self.size)#random initial weights for each feature
        self.bias = 0
    def score(self,features = []): #given vector of features, computes score from value of weights vector by taking dot product
        return np.dot(self.weights, features) + self.bias


def train_faces(PERCENT = 1,EPOCHS = 1):
    faces = read_data.read_file(fdata='facedata/facedatatrain', flabel = 'facedata/facedatatrainlabels',WIDTH = 60, HEIGHT = 70,type='faces')
    num_data =  len(faces[0])#amount of training 
    neurons = []
    hits = 0
    neurons.append(neuron(size = 60*70))
    
    for epochs in range(EPOCHS):
        faces = read_data.read_file(fdata='facedata/facedatatrain', flabel = 'facedata/facedatatrainlabels',WIDTH = 60, HEIGHT = 70,type='faces')
        for k in range(int(num_data * PERCENT)):
            x = rand.randint(0,len(faces[0])-1) #get x as random 
            features = get_features.features_from_image(faces[0][x]) #get vector of features
            
            if neurons[0].score(features) < 0 and faces[1][x] == 1:
                neurons[0].weights += features
            elif neurons[0].score(features) >= 0 and faces[1][x] == 0:
                neurons[0].weights -= features
            faces[0].pop(x)
            faces[1].pop(x)
    return neurons
    
def train_digits(PERCENT = 1,EPOCHS = 1):
    digits = read_data.read_file(fdata='digitdata/trainingimages', flabel = 'digitdata/traininglabels',WIDTH = 28, HEIGHT = 28,type='digits')
    num_data =  len(digits[0])#amount of training data
    neurons = []
    hits = 0
    for x in range(10):#create 10 neuron classes
        neurons.append(neuron(size = 28*28))
    
    for epochs in range(EPOCHS):
        digits = read_data.read_file(fdata='digitdata/trainingimages', flabel = 'digitdata/traininglabels',WIDTH = 28, HEIGHT = 28,type='digits')
        for k in range(int(num_data * PERCENT)):
            x = rand.randint(0,len(digits[0])-1) #get x as random 
            features = get_features.features_from_image(digits[0][x]) #get vector of features
            scores = []
            for y in neurons:#get score for each class
                scores.append(y.score(features))
            winnerIndex = scores.index(max(scores))
            if winnerIndex != digits[1][x]:
                neurons[winnerIndex].weights -= features
                neurons[digits[1][x]].weights += features
            digits[0].pop(x)
            digits[1].pop(x)
    return neurons

def deploy_model(PERCENT,EPOCHS, digits = True): #set digits to false to train faces
    if digits:
        data = read_data.read_file(fdata='digitdata/testimages', flabel = 'digitdata/testlabels',WIDTH = 28, HEIGHT = 28,type='digits')
    else:
        data = read_data.read_file(fdata='facedata/facedatatest', flabel = 'facedata/facedatatestlabels',WIDTH = 60, HEIGHT = 70,type='faces')
    num_data =  len(data[1])#amount of training data
    neurons = []
    if digits:
        neurons = train_digits(PERCENT = PERCENT,EPOCHS=EPOCHS)
    else:
        neurons = train_faces(PERCENT = PERCENT,EPOCHS=EPOCHS)
    hits = 0
    for x in range(num_data):
        features = get_features.features_from_image(data[0][x]) #get vector of features
        scores = []
        for y in neurons:
            scores.append(y.score(features))
        if digits:
            winnerIndex = scores.index(max(scores))
            if winnerIndex == data[1][x]:
                hits+=1
        else:
            if scores[0]<0 and data[1][x] == 0:
                hits+=1
            elif scores[0]>=0 and data[1][x] == 1:
                hits+=1
    return hits/num_data

def runTests(save = False, digits = True):
    accuracy = [] 
    accuracy.append([]) #accuracy
    accuracy.append([]) #time in seconds
    for x in range(1,11,1):
        x=x*0.1
        for y in range(1,6,1):
            start = time.time()
            acc = deploy_model(x,1,digits = digits)
            end = time.time()
            accuracy[0].append(acc)
            accuracy[1].append(end-start)
            print('Percent: %s' %x)
            print('Iter: %s' %y)
            print('Accuracy: %s' %acc)
    if save:
        if digits:
            print('Saved data to: ' + (__location__ + 'Image_classification/' + 'perceptron_digits_training_results.txt'))
            np.savetxt(__location__ + '\\perceptron_digits_training_results.txt',accuracy)
        else:
            print('Saved data to: ' + (__location__ + 'Image_classification/' + 'perceptron_faces_training_results.txt'))
            np.savetxt(__location__ + '\\perceptron_faces_training_results.txt',accuracy)

#runTests(save = True,digits=True)