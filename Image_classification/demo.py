import numpy as np
import perceptron
import naive_bayes_digits
import naive_bayes_faces
import random as rand
import matplotlib.pyplot as plt
import read_data
import get_features
import math

def bayes_digit(digit = 1,PERCENTAGE = 1):
    SMOOTHER = 1
    label_obj, num_data = naive_bayes_digits.train_digits(PERCENTAGE = 1)
    print("Trained Model!")
    digits = read_data.read_file(fdata='digitdata/testimages', flabel = 'digitdata/testlabels',WIDTH = 28, HEIGHT = 28,type='digits')


    num = 0
    while True:
        num = rand.randint(0,len(digits[0])-1)
        if digit == digits[1][num]:
            break
    print("Found Digit to Guess!")
    features = get_features.features_from_image(digits[0][num]) #get array of features
    maxls = []
    for y in  range(10):#get prob of each label and choose highest as answer
        p_y = math.log((label_obj[y].frequency) / int(num_data)*PERCENTAGE)
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
    prediction = maxls.index(max(maxls))
    print("Predicted the digit: %s" %prediction)
    plt.imshow(digits[0][num])
    plt.show()
    
def bayes_face(PERCENTAGE = 1):
    SMOOTHER = 1
    face_class,not_face_class, num_data = naive_bayes_faces.train_faces(PERCENTAGE=PERCENTAGE)
    print("Trained Model!")
    faces = read_data.read_file(fdata='facedata/facedatatest', flabel = 'facedata/facedatatestlabels',WIDTH = 60, HEIGHT = 70,type='faces')

    
    num = rand.randint(0,len(faces[0])-1)

    features = get_features.advanced_features_from_image(faces[0][num]) #get array of features
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
    prediction = maxls.index(max(maxls))


    if prediction == 0:
        print("Not Face!")
    else:
        print("Is Face!")
    plt.imshow(faces[0][num])
    plt.show()

def perceptron_deploy(digit = 1,PERCENTAGE = 1,digits = True):
    if digits:
        data = read_data.read_file(fdata='digitdata/testimages', flabel = 'digitdata/testlabels',WIDTH = 28, HEIGHT = 28,type='digits')
    else:
        data = read_data.read_file(fdata='facedata/facedatatest', flabel = 'facedata/facedatatestlabels',WIDTH = 60, HEIGHT = 70,type='faces')
    num_data =  len(data[1])#amount of training data
    neurons = []
    if digits:
        neurons = perceptron.train_digits(PERCENT = PERCENTAGE,EPOCHS=1)
    else:
        neurons = perceptron.train_faces(PERCENT = PERCENTAGE,EPOCHS=1)
    print("Trained Model!")
    hits = 0

    num = 0
    if digits:
        while True:
            num = rand.randint(0,len(data[0])-1)
            if digit == data[1][num]:
                print("Found Digit to Guess!")
                break
    else:
        num = rand.randint(0,len(data[0])-1)

        
    features = get_features.features_from_image(data[0][num]) #get vector of features
    scores = []
    for y in neurons:
        scores.append(y.score(features))
    if digits:
        winnerIndex = scores.index(max(scores))
        print("Predicted the digit: %s" %winnerIndex)
    else:
        if scores[0] < 0:
            print("Not Face!")
        else:
            print("Is Face!")
    plt.imshow(data[0][num])
    plt.show()

'''
uncomment each function below one at a time to run test on random sample from test set
'''
#bayes_digit(7, 1)
#bayes_face(1)
#perceptron_deploy(digit=8,PERCENTAGE=1,digits=True)
#perceptron_deploy(PERCENTAGE=0.1,digits=False)