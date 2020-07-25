import os
import numpy as np
import matplotlib.pyplot as plt
def read_file(fdata, flabel, type = 'digit',WIDTH = 28, HEIGHT = 28): #returns list of numpy arrays of images
    rawdatalines = []
    numpy_list = []
    numpy_list.append([]) #for image
    numpy_list.append([]) #for label
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    f_data = open(os.path.join(__location__, fdata))
    f_labels = open(os.path.join(__location__, flabel))
    if type == 'digit':
        rawdatalines = f_data.read().split('\n')
        rawlabellines = f_labels.read().split('\n')
        while rawdatalines: # as long as more lines exist
            currentdigit = np.empty([HEIGHT,WIDTH],dtype=int) 
            for y in range(HEIGHT): #put next y lines defined by HEIGHT variable into numpy array
                if rawdatalines:
                    currentline = rawdatalines.pop(0)
                else:
                    break
                for x in range(len(currentline)):
                    if currentline[x] == ' ':
                        currentdigit[y][x] = 0
                    elif currentline[x] == '+':
                        currentdigit[y][x] = 1
                    elif currentline[x] == '#':
                        currentdigit[y][x] = 2
            numpy_list[0].append(currentdigit)
            label = rawlabellines.pop(0)
            if label.isdigit():
                numpy_list[1].append(int(label))
    return numpy_list 

digits = read_file(fdata='digitdata/testimages', flabel = 'digitdata/testlabels',type = 'digit',WIDTH = 28, HEIGHT = 28)
print(digits[1][0])
plt.imshow(digits[0][0], interpolation='nearest')
plt.show()