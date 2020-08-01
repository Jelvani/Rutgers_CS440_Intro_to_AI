import matplotlib.pyplot as plt
import read_data
import get_features
import numpy as np
import math
import random as rand
import os


class label():#represents a class of face
    v0 = []
    v1 = []
    frequency = 0 #amount of times label is seen in training

def train_faces(PERCENTAGE = 1):
    
    faces = read_data.read_file(fdata='facedata/facedatatrain', flabel = 'facedata/facedatatrainlabels',WIDTH = 60, HEIGHT = 74,type='faces')
    num_data =  len(faces[0])#amount of training data
    features = get_features.features_from_image(faces[0][0])
    face_class = label()
    face_class.v0 = np.zeros(len(features))
    face_class.v1 = np.zeros(len(features))
    not_face_class = label()
    not_face_class.v0 = np.zeros(len(features))
    not_face_class.v1 = np.zeros(len(features))
    '''
    get frequency of feature values for each feature in training set
    '''
    for k in range(int(num_data * PERCENTAGE)): # for each training data number
        x = rand.randint(0,len(faces[0])-1) #get x as random index
        features = get_features.features_from_image(faces[0][x]) #get vector of features

        if faces[1][x] == 0:
            not_face_class.frequency+=1
        elif faces[1][x] == 1:
            face_class.frequency+=1
        for y in range(len(features)):
            if faces[1][x] == 0:
                if features[y]==0:
                    not_face_class.v0[y]+=1
                elif features[y]==1:
                    not_face_class.v1[y]+=1
            elif faces[1][x] == 1:
                if features[y]==0:
                    face_class.v0[y]+=1
                elif features[y]==1:
                    face_class.v1[y]+=1
        faces[0].pop(x)
        faces[1].pop(x)

    '''
    Now we will compute the posterior given by MAX{p(label | features) = p(features | label) * p(label)}
    '''
    faces = read_data.read_file(fdata='facedata/facedatatest', flabel = 'facedata/facedatatestlabels',WIDTH = 60, HEIGHT = 74,type='faces')
    predictions = [] #outputs from bayes classifier
    
    for x in range(len(faces[0])):
        features = get_features.features_from_image(faces[0][x]) #get array of features
        maxls = []
        cur_guess = None

        'compute probabilties for a face'
        p_y = math.log((face_class.frequency+1) / int(num_data*PERCENTAGE))
        likelihood = 0
        for feats in range(len(features)):
            if features[feats]==0:
                likelihood+= math.log((face_class.v0[feats]+1)/(face_class.frequency+1))
            elif features[feats]==1:
                likelihood+= math.log((face_class.v1[feats]+1)/(face_class.frequency+1))
        likelihood = likelihood + p_y
        maxls.append(likelihood)

        p_y = math.log((not_face_class.frequency+1) / int(num_data*PERCENTAGE))
        likelihood = 0
        for feats in range(len(features)):
            if features[feats]==0:
                likelihood+= math.log((not_face_class.v0[feats]+1)/(not_face_class.frequency+1))
            elif features[feats]==1:
                likelihood+= math.log((not_face_class.v1[feats]+1)/(not_face_class.frequency+1))
        likelihood = likelihood + p_y
        maxls.append(likelihood)

        predictions.append(maxls.index(max(maxls)))

    hits = 0
    for x in predictions:
        if predictions[x] == faces[1][x]:
            hits+=1
    accuracy = hits/len(faces[1])
    print("Accuracy of: %s" %(accuracy))
    return accuracy



acc = []
for x in range(1,10,1):
    for y in range(1,5,1):
        acc.append(train_faces(PERCENTAGE = x/10))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
np.savetxt(__location__ + 'faces.txt',acc)
h1 = plt.plot(acc)
plt.show(h1)
