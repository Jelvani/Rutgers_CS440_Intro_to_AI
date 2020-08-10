import matplotlib.pyplot as plt
import read_data
import get_features
import numpy as np
import math
import random as rand
import os
import time

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
class label():#represents a class of digit
    v0 = []
    v1 = []
    v2 = []
    frequency = 1 #amount of times label is seen in training

def train_digits(PERCENTAGE = 1):
    
    digits = read_data.read_file(fdata='digitdata/trainingimages', flabel = 'digitdata/traininglabels',WIDTH = 28, HEIGHT = 28,type='digits')
    num_data =  len(digits[0])#amount of training data
    label_obj = []
    features = get_features.features_from_image(digits[0][0])
    for x in range(10): #create 10 label objects for each class 
        lbl = label()
        lbl.v0 = np.ones(len(features))
        lbl.v1 = np.ones(len(features))
        lbl.v2 = np.ones(len(features))
        label_obj.append(lbl)
    '''
    get frequency of feature values for each feature in training set
    '''
    for k in range(int(num_data * PERCENTAGE)): # for each training data number
        x = rand.randint(0,len(digits[0])-1) #get x as random index
        features = get_features.features_from_image(digits[0][x]) #get vector of features
        label_obj[digits[1][x]].frequency+=1
        for y in range(len(features)):
            if features[y]==0:
                label_obj[digits[1][x]].v0[y]+=1
            elif features[y]==1:
                label_obj[digits[1][x]].v1[y]+=1
            elif features[y]==2:
                label_obj[digits[1][x]].v2[y]+=1
        digits[0].pop(x)
        digits[1].pop(x)

    return label_obj, num_data


'''
    Now we will compute the posterior given by MAX{p(label | features) = p(features | label) * p(label)}
    '''
def infrence_model(PERCENTAGE = 1):
    SMOOTHER = 1
    label_obj, num_data = train_digits(PERCENTAGE = PERCENTAGE)
    
    digits = read_data.read_file(fdata='digitdata/testimages', flabel = 'digitdata/testlabels',WIDTH = 28, HEIGHT = 28,type='digits')

    predictions = [] #outputs from bayes classifier
    for x in range(len(digits[0])):
        features = get_features.features_from_image(digits[0][x]) #get array of features
        maxls = []
        cur_guess = None
        for y in  range(10):#get prob of each label and choose highest as answer
            p_y = math.log((label_obj[y].frequency) / int(num_data*PERCENTAGE))
            likelihood = 0
            for feats in range(len(features)):
                if features[feats]==0:
                    likelihood+= math.log((label_obj[y].v0[feats] + SMOOTHER)/(label_obj[y].frequency +label_obj[y].v0[feats])*SMOOTHER)
                elif features[feats]==1:
                    likelihood+= math.log((label_obj[y].v1[feats] + SMOOTHER)/(label_obj[y].frequency + label_obj[y].v1[feats])*SMOOTHER)
                elif features[feats]==2:
                    likelihood+= math.log((label_obj[y].v2[feats] + SMOOTHER)/(label_obj[y].frequency + label_obj[y].v2[feats])*SMOOTHER)    
            likelihood = likelihood + p_y
            maxls.append(likelihood)
        predictions.append(maxls.index(max(maxls)))

    hits = 0
    for x in range(len(digits[1])):
        if predictions[x] == digits[1][x]:
            hits+=1
    accuracy = hits/len(digits[1])
    return accuracy

def runTests(save = False):
    accuracy = [] 
    accuracy.append([]) #accuracy
    accuracy.append([]) #time in seconds
    for x in range(1,11,1):
        x=x*0.1
        for y in range(1,6,1):
            start = time.time()
            acc = infrence_model(PERCENTAGE = x)
            end = time.time()
            accuracy[0].append(acc)
            accuracy[1].append(end-start)
            print('Percent: %s' %x)
            print('Iter: %s' %y)
            print('Accuracy: %s' %acc)
    if save:
        print('Saved data to: ' + (__location__ + 'Image_classification/' + 'bayes_digits_training_results.txt'))
        np.savetxt(__location__ + '\\bayes_digits_training_results.txt',accuracy)

#runTests(save = False)