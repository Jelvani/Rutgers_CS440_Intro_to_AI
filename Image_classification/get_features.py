import numpy as np
'''
Used for extracting features from a given numpyarray image
Returns a list of features
'''

def features_from_image(image):
    return image.flatten()

def advanced_features_from_image(image):
    lines = []
    for x in range(0,len(image)-5,5):
        for y in range(0,len(image[x])-5,5):
            sum = 0
            for z in range(5):
                for k in range(5):
                    sum+=image[x+z][y+k]
            if sum<8:
                lines.append(1)
            else:
                lines.append(0)
    return lines