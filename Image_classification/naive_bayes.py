import matplotlib.pyplot as plt
import read_data
import get_features
import numpy as np
import math
class label():
    feature_prob = [] #list of probabilties for each feature
    frequency = 0 #amount of times label is seen in training

#digits = read_data.read_file(fdata='facedata/facedatatest', flabel = 'facedata/facedatatestlabels',WIDTH = 60, HEIGHT = 74)
digits = read_data.read_file(fdata='digitdata/trainingimages', flabel = 'digitdata/traininglabels',WIDTH = 28, HEIGHT = 28)
num_data =  len(digits[0])-1#amount of training data
label_obj = []
features = get_features.features_from_image(digits[0][0])
for x in range(10): #create 10 label objects for each class 
    lbl = label()
    lbl.feature_prob = np.zeros(len(features))
    label_obj.append(lbl)

for x in range(num_data): # for each training data number
    features = get_features.features_from_image(digits[0][x]) #get array of feature averages
    label_obj[digits[1][x]].frequency+=1 #increment frequency of label
    for y in range(len(features)): #for each feature in image
        label_obj[digits[1][x]].feature_prob[y]+=features[y]

for x in range(10):#compute average features for each class
    for y in range(len(features)):
        label_obj[x].feature_prob[y] = label_obj[x].feature_prob[y]/label_obj[x].frequency

'''
Now we will compute the posterior given by MAX{p(label | features) = p(features | label) * p(label)}
'''
digits = read_data.read_file(fdata='digitdata/testimages', flabel = 'digitdata/testlabels',WIDTH = 28, HEIGHT = 28)

predictions = [] #outputs from bayes classifier
for x in range(len(digits[0])-1):
    features = get_features.features_from_image(digits[0][x]) #get array of feature averages
    maxls = []
    cur_guess = None
    for y in  range(10):#get prob of each label and choose highest as answer
        p_y = math.log(label_obj[y].frequency / num_data)
        likelihood = 0
        for feats in range(len(features)):
            likelihood+= (features[feats] * label_obj[y].feature_prob[feats])
        likelihood = likelihood + p_y
        maxls.append(likelihood)
    predictions.append(maxls.index(min(maxls)))

hits = 0
for x in predictions:
    if predictions[x] == digits[1][x]:
        hits+=1
accuracy = hits/len(digits[1])
print("Accuracy of: %s" %(accuracy))