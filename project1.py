import math
import os
import random
from string import punctuation, digits

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_order(n_samples):
    """Return a deterministic or random ordering of sample indices."""
    order_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              str(n_samples) + '.txt')
    try:
        with open(order_file) as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


# ---------------------------------------------------------------------------
# Part I – Loss functions
# ---------------------------------------------------------------------------

def hinge_loss_single(feature_vector, label, theta, theta_0):
    """Hinge loss on a single data point."""
    y_pred = np.dot(theta.transpose(), feature_vector) + theta_0
    return max(0, 1 - np.dot(label, y_pred))


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """Average hinge loss over a dataset."""
    losses = []
    for i in range(labels.size):
        y_pred = np.dot(theta, feature_matrix[i]) + theta_0
        losses.append(max(0, 1 - np.dot(labels[i], y_pred)))
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Part I – Classifiers
# ---------------------------------------------------------------------------

def perceptron_single_step_update(feature_vector, label, current_theta,
                                  current_theta_0):
    """Single step of the perceptron update rule."""
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
    Full perceptron algorithm.

    Runs T iterations through the data set using the ordering returned by
    ``get_order``.
    """
    current_theta_0 = 0.0
    current_theta = np.zeros(feature_matrix.shape[1])
    for _t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            current_theta, current_theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i],
                current_theta, current_theta_0)
    return (current_theta, current_theta_0)


def average_perceptron(feature_matrix, labels, T):
    """
    Average perceptron algorithm.

    Runs T iterations and returns the average of all intermediate
    (theta, theta_0) values.
    """
    n_samples = feature_matrix.shape[0]
    nt = n_samples * T
    current_theta_0 = 0.0
    current_theta = np.zeros(feature_matrix.shape[1])
    theta_accumulator = 0
    theta_0_accumulator = 0
    for _t in range(T):
        for i in get_order(n_samples):
            current_theta, current_theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i],
                current_theta, current_theta_0)
            theta_accumulator = theta_accumulator + current_theta
            theta_0_accumulator += current_theta_0

    return (theta_accumulator / nt, theta_0_accumulator / nt)


def pegasos_single_step_update(feature_vector, label, L, eta,
                               current_theta, current_theta_0):
    """Single step of the Pegasos (SVM) update rule."""
    y_pred = np.dot(current_theta, feature_vector) + current_theta_0

    if np.dot(label, y_pred) <= 1:
        current_theta = (1 - eta * L) * current_theta + eta * label * feature_vector
        current_theta_0 += eta * label
    else:
        current_theta = (1 - eta * L) * current_theta

    return (current_theta, current_theta_0)


def pegasos(feature_matrix, labels, T, L):
    """
    Pegasos (Primal Estimated sub-GrAdient SOlver for SVM).

    Learning rate = 1 / sqrt(t) where t counts total updates performed.
    """
    current_theta_0 = 0.0
    k = 1
    current_theta = np.zeros(feature_matrix.shape[1])
    for _t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            eta = 1 / math.sqrt(k)
            current_theta, current_theta_0 = pegasos_single_step_update(
                feature_matrix[i], labels[i],
                L, eta, current_theta, current_theta_0)
            k += 1
    return (current_theta, current_theta_0)


# ---------------------------------------------------------------------------
# Part II – Feature extraction & classification
# ---------------------------------------------------------------------------

def classify(feature_matrix, theta, theta_0):
    """
    Classify data points.

    Returns +1 for predictions > 0, else -1.
    """
    predictions = np.dot(feature_matrix, theta) + theta_0
    return np.where(predictions > 0, 1.0, -1.0)


def classifier_accuracy(classifier, train_feature_matrix, val_feature_matrix,
                         train_labels, val_labels, **kwargs):
    """
    Train a classifier and return (train_accuracy, val_accuracy).
    """
    train_t, train_t0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_ypred = classify(train_feature_matrix, train_t, train_t0)
    val_ypred = classify(val_feature_matrix, train_t, train_t0)
    return (accuracy(train_ypred, train_labels), accuracy(val_ypred, val_labels))


def extract_words(input_string):
    """
    Tokenise a string: lowercase, separate punctuation and digits into
    their own tokens.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def bag_of_words(texts):
    """
    Build a word-to-index dictionary from *texts*, excluding stopwords.
    """
    stopwords_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "stopwords.txt")
    with open(stopwords_path, "r") as f:
        stopwords = {line.strip() for line in f}

    dictionary = {}
    for text in texts:
        for word in extract_words(text):
            if word not in dictionary and word not in stopwords:
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Return a (n_reviews × vocab_size) bag-of-words feature matrix.

    Uses word counts (not binary indicators).
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
    """Fraction of correct predictions."""
    return (preds == targets).mean()
