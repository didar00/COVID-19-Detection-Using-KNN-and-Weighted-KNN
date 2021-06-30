'''

            WEIGHTED K NEAREST NEIGHBOR ALGORITHM


'''


''' imports '''
from random import randrange
from csv import reader
from filters import *
from math import sqrt
import os





'''
Calculate the Euclidean distance
between two vectors of data
'''
def euclidean_distance(arr1, arr2):
    distance = 0.0
    for i in range(1, len(arr2)-1):
        distance += np.sum(np.square(arr1[i]-arr2[i]))
    return np.sqrt(distance)



'''
Cross validation method to split a data into n folds,
return grouped data inside a list
'''
def cross_validation_split(data, n_folds):
    data_split = list()
    data_copy = list(data)
    fold_size = int(len(data) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(data_copy))
            fold.append(data_copy.pop(index))
        data_split.append(fold)
    return data_split


'''
Runs knn algorithm through applying cross validation
on training data in case test dataset is not provided
'''
def weighted_knn_with_CV(dataset, n_folds,k):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = weighted_knn(train_set, test_set, k)
        actual = [row[-1] for row in fold]
        accuracy = get_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


'''
Runs weighted knn algorithm with the provided test dataset
'''
def weighted_knn(train_set, test_set, k):
    scores = list()
    predicted = weighted_knn_helper(train_set, test_set, k)
    actual = [row[-1] for row in test_set]
    accuracy = get_accuracy(actual, predicted)
    scores.append(accuracy)
    return scores


'''
Weighted K Nearest Neighbor algorithm to classify
the image according to the k neighbors' weights
'''
def weighted_knn_helper(train, test, k):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, k)
        predictions.append(output)
    return(predictions)


'''
Predict label of the image through using
inverse distance weighting method
'''
def predict_classification(train, test_row, k):
    neighbors = get_neighbors(train, test_row, k)
    output_values = [row[-1] for row in neighbors]

    label1_freq = 0.0
    label2_freq = 0.0
    label3_freq = 0.0
    for neighbor in neighbors:
        if neighbor[1] != 0:
            if neighbor[0][-1] == "COVID":
                label1_freq += 1/neighbor[1]
            elif neighbor[0][-1] == "NORMAL":
                label2_freq += 1/neighbor[1]
            elif neighbor[0][-1] == "Viral Pneumonia":
                label3_freq += 1/neighbor[1]
        
    if label1_freq > label2_freq and label1_freq > label3_freq:
        return "COVID"   
    elif label2_freq > label1_freq and label2_freq > label3_freq:
        return "NORMAL" 
    elif label3_freq > label1_freq and label3_freq > label2_freq:
        return "Viral Pneumonia"


'''
Return the list of nearest k neighbors of test_row
after obtaining and sorting the distances
between test data and all training data
'''
def get_neighbors(train, test_row, k):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i])
    return neighbors


'''
Calculate the accuracy of classification
through calculating the actual and predicted
result
'''
def get_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0