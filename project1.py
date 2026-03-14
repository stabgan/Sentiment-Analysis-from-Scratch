import math
from string import punctuation, digits

import numpy as np
import random


# Part I


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    y_pred = np.dot(theta.transpose(), feature_vector) + theta_0
    return max(0, 1 - np.dot(label, y_pred))


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    losses = []
    for i in range(labels.size):
        y_pred = np.dot(theta, feature_matrix[i]) + theta_0
        losses.append(max(0, 1 - np.dot(labels[i], y_pred)))
    return sum(losses) / len(losses)


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    y_pred = np.dot(current_theta, feature_vector) + current_theta_0

    if np.dot(label, y_pred) <= 0:
        theta = current_theta + np.dot(label, feature_vector)
        theta_0 = current_theta_0 + label
    else:
        theta = current_theta
        theta_0 = current_theta_0

    return (theta, theta_0)


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    current_theta_0 = 0.0
    current_theta = np.zeros(feature_matrix.shape[1])
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            current_theta, current_theta_0 = perceptron_single_step_update(
                feature_matrix[i],
                labels[i],
                current_theta,
                current_theta_0)
    return (current_theta, current_theta_0)


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    nt = feature_matrix.shape[0] * T
    current_theta_0 = 0.0
    current_theta = np.zeros(feature_matrix.shape[1])
    theta_accumulator = 0
    theta_0_accumulator = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            current_theta, current_theta_0 = perceptron_single_step_update(
                feature_matrix[i],
                labels[i],
                current_theta,
                current_theta_0)
            theta_accumulator = theta_accumulator + current_theta
            theta_0_accumulator += current_theta_0

    return (theta_accumulator / nt, theta_0_accumulator / nt)


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    y_pred = np.dot(current_theta, feature_vector) + current_theta_0

    if np.dot(label, y_pred) <= 1:
        current_theta = (1 - eta * L) * current_theta + eta * label * feature_vector
        current_theta_0 += eta * label
    else:
        current_theta = (1 - eta * L) * current_theta

    return (current_theta, current_theta_0)


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    current_theta_0 = 0.0
    k = 1
    current_theta = np.zeros(feature_matrix.shape[1])
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            eta = 1 / math.sqrt(k)
            current_theta, current_theta_0 = pegasos_single_step_update(
                feature_matrix[i],
                labels[i],
                L,
                eta,
                current_theta,
                current_theta_0)
            k += 1
    return (current_theta, current_theta_0)


# Part II


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    predictions = []
    for i in range(feature_matrix.shape[0]):
        y_pred = np.dot(theta.transpose(), feature_matrix[i]) + theta_0
        if y_pred > 0:
            predictions.append(1)
        else:
            predictions.append(-1)
    return np.array(predictions, dtype="float64")


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and returns its accuracy on train and validation data.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs)
        train_feature_matrix - A numpy matrix describing the training data.
        val_feature_matrix - A numpy matrix describing the validation data.
        train_labels - A numpy array of correct training labels.
        val_labels - A numpy array of correct validation labels.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    train_t, train_t0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_ypred = classify(train_feature_matrix, train_t, train_t0)
    val_ypred = classify(val_feature_matrix, train_t, train_t0)

    return (accuracy(train_ypred, train_labels), accuracy(val_ypred, val_labels))


def extract_words(input_string):
    """
    Helper function for bag_of_words().
    Inputs a text string.
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews.
    Returns a dictionary of unique unigrams occurring over the input,
    excluding stopwords loaded from stopwords.txt.
    """
    with open("stopwords.txt", "r") as f:
        stopwords = [line.strip() for line in f]

    dictionary = {}
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stopwords:
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews.
    Inputs the dictionary of words as given by bag_of_words.
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = word_list.count(word)
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
