import numpy as np
'''
Used for extracting features from a given numpyarray image
Returns a list of features
'''

def features_from_image(image):
    return image.flatten()