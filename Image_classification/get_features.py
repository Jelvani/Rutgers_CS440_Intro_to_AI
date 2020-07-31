import numpy as np
'''
Used for extracting features from a given numpyarray image
Returns a list of features
'''

def features_from_image(image):
    features = []
    features = image.mean(axis = 1) #average each row of image pixels
    return features #list containing average of each row