import matplotlib.pyplot as plt
import read_data
import get_features
import numpy as np
import math
import random as rand
import os
import time

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
class label():#represents a class of face
    features = []
    frequency = 1 #amount of times label is seen in training

def train_faces(PERCENTAGE = 1):
    
    faces = read_data.read_file(fdata='facedata/facedatatrain', flabel = 'facedata/facedatatrainlabels',WIDTH = 60, HEIGHT = 70,type='faces')
    num_data =  len(faces[0])#amount of training data
    features = get_features.advanced_features_from_image(faces[0][0])
    face_class = label()
    face_class.features = np.zeros(len(features))
    not_face_class = label()
    not_face_class.features = np.zeros(len(features))
    '''
    get frequency of feature values for each feature in training set
    '''
    for k in range(int(num_data * PERCENTAGE)): # for each training data number
        x = rand.randint(0,len(faces[0])-1) #get x as random index
        features = get_features.advanced_features_from_image(faces[0][x]) #get vector of features
        if faces[1][x] == 0:
            not_face_class.frequency+=1
            not_face_class.features+=features
        elif faces[1][x] == 1:
            face_class.frequency+=1
            face_class.features+=features
        faces[0].pop(x)
        faces[1].pop(x)
    return face_class,not_face_class, num_data


def infrence_model(PERCENTAGE = 1):
    '''
    Now we will compute the posterior given by MAX{p(label | features) = p(features | label) * p(label)}
    '''
    SMOOTHER = 1
    face_class,not_face_class, num_data = train_faces(PERCENTAGE=PERCENTAGE)
    faces = read_data.read_file(fdata='facedata/facedatatest', flabel = 'facedata/facedatatestlabels',WIDTH = 60, HEIGHT = 70,type='faces')
    predictions = [] #outputs from bayes classifier
    
    for x in range(len(faces[0])):
        features = get_features.advanced_features_from_image(faces[0][x]) #get array of features
        maxls = []

        'compute probabilties for not a face'
        p_y = math.log((not_face_class.frequency) / int(num_data*PERCENTAGE))
        likelihood = 0
        for feats in range(len(features)):
            if features[feats]==0:
                likelihood+= math.log(((not_face_class.frequency-not_face_class.features[feats])+SMOOTHER)/(not_face_class.frequency + (not_face_class.frequency-not_face_class.features[feats]))*SMOOTHER)
            elif features[feats]==1:
                likelihood+= math.log((not_face_class.features[feats]+SMOOTHER)/(not_face_class.frequency + not_face_class.features[feats])*SMOOTHER)
        likelihood = likelihood + p_y
        maxls.append(likelihood)
        'compute probabilties for a face'

        p_y = math.log((face_class.frequency) / int(num_data*PERCENTAGE))
        likelihood = 0
        for feats in range(len(features)):
            if features[feats]==0:
                likelihood+= math.log(((face_class.frequency-face_class.features[feats])+SMOOTHER)/(face_class.frequency + (face_class.frequency-face_class.features[feats]))*SMOOTHER)
            elif features[feats]==1:
                likelihood+= math.log((face_class.features[feats]+SMOOTHER)/(face_class.frequency + face_class.features[feats])*SMOOTHER)
        likelihood = likelihood + p_y
        maxls.append(likelihood)
        predictions.append(maxls.index(max(maxls)))

    hits = 0
    for x in range(len(faces[1])):
        if predictions[x] == faces[1][x]:
            hits+=1
    accuracy = hits/len(faces[1])
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
        print('Saved data to: ' + (__location__ + 'Image_classification/' + 'bayes_faces_training_results.txt'))
        np.savetxt(__location__ + '\\bayes_faces_training_results.txt',accuracy)

#runTests(False)