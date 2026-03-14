"""
Sentiment Analysis – main driver script.

Loads review data, builds bag-of-words features, trains a Pegasos SVM
classifier, and reports test accuracy.
"""

import os

import numpy as np

import project1 as p1
import utils

# ---------------------------------------------------------------------------
# Resolve data paths relative to this script
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))


def _data_path(filename):
    return os.path.join(_DIR, filename)


def main():
    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    train_data = utils.load_data(_data_path('reviews_train.tsv'))
    val_data = utils.load_data(_data_path('reviews_val.tsv'))
    test_data = utils.load_data(_data_path('reviews_test.tsv'))

    train_texts, train_labels = zip(
        *((s['text'], s['sentiment']) for s in train_data))
    val_texts, val_labels = zip(
        *((s['text'], s['sentiment']) for s in val_data))
    test_texts, test_labels = zip(
        *((s['text'], s['sentiment']) for s in test_data))

    dictionary = p1.bag_of_words(train_texts)

    train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
    val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
    test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)

    # ------------------------------------------------------------------
    # Evaluate best model (Pegasos, T=25, L=0.01) on the test set
    # ------------------------------------------------------------------
    _train_acc, test_acc = p1.classifier_accuracy(
        p1.pegasos,
        train_bow_features,
        test_bow_features,
        train_labels,
        test_labels,
        T=25, L=0.01,
    )
    print(f"Test accuracy (Pegasos T=25 L=0.01): {test_acc:.4f}")


if __name__ == "__main__":
    main()
