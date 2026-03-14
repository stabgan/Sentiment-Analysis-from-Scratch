import csv

import matplotlib.pyplot as plt
import numpy as np

import project1 as p1


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_toy_data(path_toy_data):
    """Load the 2-D toy dataset as (features, labels)."""
    labels, xs, ys = np.loadtxt(path_toy_data, delimiter='\t', unpack=True)
    return np.vstack((xs, ys)).T, labels


def load_data(path_data, extras=False):
    """
    Load a TSV review dataset.

    Returns a list of dicts with keys *sentiment* (+1 / -1) and *text*.
    If *extras* is True the dicts also contain productId, userId, summary,
    helpfulY, and helpfulN.
    """
    basic_fields = {'sentiment', 'text'}
    numeric_fields = {'sentiment', 'helpfulY', 'helpfulN'}

    data = []
    with open(path_data, encoding="latin1") as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            for field in list(datum.keys()):
                if not extras and field not in basic_fields:
                    del datum[field]
                elif field in numeric_fields and datum[field]:
                    datum[field] = int(datum[field])
            data.append(datum)
    return data


def write_predictions(path_submit_data, preds):
    """Write predicted sentiments back into a TSV file."""
    with open(path_submit_data, encoding="latin1") as f_data:
        reader = csv.DictReader(f_data, delimiter='\t')
        data = list(reader)
        fieldnames = reader.fieldnames

    assert len(preds) == len(data), \
        'Expected {} predictions but {} were given.'.format(len(data), len(preds))

    for pred, datum in zip(preds.astype(int), data):
        assert pred == 1 or pred == -1, 'Invalid prediction: {}.'.format(pred)
        datum['sentiment'] = pred

    with open(path_submit_data, 'w', newline='') as f_out:
        writer = csv.DictWriter(f_out, delimiter='\t', fieldnames=fieldnames)
        writer.writeheader()
        for datum in data:
            writer.writerow(datum)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_toy_data(algo_name, features, labels, thetas):
    """Plot 2-D toy data with the decision boundary."""
    plt.subplots()
    colors = ['b' if label == 1 else 'r' for label in labels]
    plt.scatter(features[:, 0], features[:, 1], s=40, c=colors)
    xmin, xmax = plt.axis()[:2]

    theta, theta_0 = thetas
    xs = np.linspace(xmin, xmax)
    ys = -(theta[0] * xs + theta_0) / (theta[1] + 1e-16)
    plt.plot(xs, ys, 'k-')

    title = ' '.join(word.capitalize() for word in algo_name.split())
    plt.suptitle('Classified Toy Data ({})'.format(title))
    plt.show()


def plot_tune_results(algo_name, param_name, param_vals, acc_train, acc_val):
    """Plot train/val accuracy vs. a hyperparameter."""
    plt.subplots()
    plt.plot(param_vals, acc_train, '-o')
    plt.plot(param_vals, acc_val, '-o')

    title = ' '.join(word.capitalize() for word in algo_name.split())
    plt.suptitle('Classification Accuracy vs {} ({})'.format(
        param_name.capitalize(), title))
    plt.legend(['train', 'val'], loc='upper right', title='Partition')
    plt.xlabel(param_name.capitalize())
    plt.ylabel('Accuracy (%)')
    plt.show()


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune(train_fn, param_vals, train_feats, train_labels, val_feats,
         val_labels):
    """Evaluate *train_fn* over a list of hyperparameter values."""
    train_accs = np.ndarray(len(param_vals))
    val_accs = np.ndarray(len(param_vals))

    for i, val in enumerate(param_vals):
        theta, theta_0 = train_fn(train_feats, train_labels, val)
        train_preds = p1.classify(train_feats, theta, theta_0)
        train_accs[i] = p1.accuracy(train_preds, train_labels)
        val_preds = p1.classify(val_feats, theta, theta_0)
        val_accs[i] = p1.accuracy(val_preds, val_labels)

    return train_accs, val_accs


def tune_perceptron(*args):
    return tune(p1.perceptron, *args)


def tune_avg_perceptron(*args):
    return tune(p1.average_perceptron, *args)


def tune_pegasos_T(best_L, *args):
    def train_fn(features, labels, T):
        return p1.pegasos(features, labels, T, best_L)
    return tune(train_fn, *args)


def tune_pegasos_L(best_T, *args):
    def train_fn(features, labels, L):
        return p1.pegasos(features, labels, best_T, L)
    return tune(train_fn, *args)


def most_explanatory_word(theta, wordlist):
    """Return words sorted by descending theta weight."""
    return [word for (_, word) in sorted(zip(theta, wordlist), reverse=True)]
