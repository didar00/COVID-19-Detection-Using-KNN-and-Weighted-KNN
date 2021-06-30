'''
        MAIN PROGRAM TO RUN TWO DIFFERENT ALGORITHMS
                    KNN AND WEIGHTED_KNN

'''

import matplotlib.pyplot as plt
from random import randrange
from csv import reader
from math import sqrt
import time
import os

'''
import modules
'''
from weighted_knn import *
from filters import *
from knn import *

'''

PLEASE DETERMINE THE SIZE OF THE
TRAINING AND SET DATA

'''
TRAINING_SIZE = 12
TEST_SIZE = 4


training_data = list()
test_data = list()

'''
dimensions to resize the image
to smaller size
'''
dim = (100,100)

'''
Read images and obtain their features 
through using Canny edge detection and
Gabor filter
'''
i = 0
images = os.listdir('train/COVID')
for image in images:
    img = cv2.imread("train/COVID/"+image)
    # resize the image with predefined dimensions
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # apply Gabor filter to the image
    out = Gabor_process(img)
    # turn filtered image into vector for image processing
    gabor = np.reshape(out, -1)
    # apply Canny edge detection
    out = Canny_edge(img)
    # turn it into vector
    canny = np.reshape(out, -1)
    training_data.append((image, canny, gabor, "COVID"))
    i+=1
    if i > TRAINING_SIZE/3:
        break;


i = 0
images = os.listdir("train/NORMAL")
for image in images:
    img = cv2.imread("train/NORMAL/"+image)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    out = Gabor_process(img)
    gabor = np.reshape(out, -1)
    out = Canny_edge(img)
    canny = np.reshape(out, -1)
    training_data.append((image, canny, gabor, "NORMAL"))
    i+=1
    if i > TRAINING_SIZE/3:
        break;


i = 0
images = os.listdir("train/Viral Pneumonia")
for image in images:
    img = cv2.imread("train/Viral Pneumonia/"+image)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    out = Gabor_process(img)
    gabor = np.reshape(out, -1)
    out = Canny_edge(img)
    canny = np.reshape(out, -1)
    training_data.append((image, canny, gabor, "Viral Pneumonia"))
    i+=1
    if i > TRAINING_SIZE/3:
        break;

'''
READ TEST DATA
'''
i = 0
images = os.listdir("test")
for image in images:
    img = cv2.imread("test/"+image)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    out = Gabor_process(img)
    gabor = np.reshape(out, -1)
    out = Canny_edge(img)
    canny = np.reshape(out, -1)
    name = image.split(" ")[0]
    test_data.append((image, canny, gabor, name))
    i+=1
    if i > TEST_SIZE:
        break;


print("we are there")

n_folds = 5 # number of folds used in croos-validation
k_values = [1, 3, 5, 7, 9, 11]
accuracy_list = []
for k in k_values:
    '''
    Call "knn" to run KNN algorithm
    Call "weighted_knn" function for weighted KNN algorithm
    '''
    scores = knn(training_data, test_data, k)
    #scores = weighted_knn(training_data, test_data, k)
    accuracy_list.append(sum(scores)/float(len(scores)))


'''
Print accuracy values for each k value
and show mean accuracy
'''
for i, acc in zip(k_values, accuracy_list):
    print("k value = ", i, "    Accuracy = ", acc)

print("Mean Accuracy: %.3f%%" % (sum(accuracy_list)/float(len(accuracy_list))))

'''
Plot the accuracy results
'''
plt.xlabel("k neigbors")
plt.ylabel("Accuracy")
plt.bar(k_values, accuracy_list)
plt.tight_layout()
plt.show()
