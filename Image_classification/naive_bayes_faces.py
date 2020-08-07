import matplotlib.pyplot as plt
import read_data
import get_features
import numpy as np
import math
import random as rand
import os


class label():#represents a class of face
    features = []
    frequency = 1 #amount of times label is seen in training

def train_faces(PERCENTAGE = 1):
    SMOOTHER = 1
    faces = read_data.read_file(fdata='facedata/facedatatrain', flabel = 'facedata/facedatatrainlabels',WIDTH = 60, HEIGHT = 74,type='faces')
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


    '''
    Now we will compute the posterior given by MAX{p(label | features) = p(features | label) * p(label)}
    '''
    faces = read_data.read_file(fdata='facedata/facedatavalidation', flabel = 'facedata/facedatavalidationlabels',WIDTH = 60, HEIGHT = 74,type='faces')
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
    print("Accuracy of: %s" %(accuracy))
    return accuracy

acc = []
for x in range(1,10,1):
    for y in range(1,5,1):
        acc.append(train_faces(PERCENTAGE = 10/10))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
np.savetxt(__location__ + 'faces.txt',acc)
h1 = plt.plot(acc)
plt.show(h1)
