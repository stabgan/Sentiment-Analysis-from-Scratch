# Sentiment Analysis from Scratch

Classify product reviews as positive or negative using linear classifiers implemented entirely in NumPy.

## What It Does

Trains three linear classifiers on bag-of-words features extracted from product reviews:

| Algorithm | Description |
|-----------|-------------|
| Perceptron | Classic online linear classifier |
| Average Perceptron | Smoothed variant that averages weight vectors across all updates |
| Pegasos (SVM) | Primal Estimated sub-GrAdient SOlver for SVM |

The pipeline tokenises review text, removes stopwords, builds a word-count feature matrix, and trains each classifier with configurable hyperparameters (iterations T, regularisation L).

## Dataset

Tab-separated review files included in the repo:

| File | Purpose |
|------|---------|
| reviews_train.tsv | Training set |
| reviews_val.tsv | Validation and hyperparameter tuning |
| reviews_test.tsv | Held-out test evaluation |
| toy_data.tsv | Tiny 2-D dataset for visualisation |

Each row contains a sentiment label (+1 or -1) and the review text.

## Tech Stack

| | Technology | Role |
|---|-----------|------|
| Python | Python 3.8+ | Language |
| NumPy | NumPy | Linear algebra and feature matrices |
| Matplotlib | Matplotlib | Plotting decision boundaries and tuning curves |

## Getting Started

```bash
git clone https://github.com/stabgan/Sentiment-Analysis-from-Scratch.git
cd Sentiment-Analysis-from-Scratch

pip install -r requirements.txt

# Run the test suite
python test.py

# Train and evaluate on the full dataset
python main.py
```

## Project Structure

```
project1.py        Core algorithms (perceptron, pegasos, BoW features)
utils.py           Data loading, plotting, hyperparameter tuning
main.py            Driver script: loads data, trains, reports accuracy
test.py            Unit tests for every algorithm
stopwords.txt      Stopword list for feature filtering
requirements.txt   Python dependencies
```

## References

- Pegasos: Shalev-Shwartz et al., 2007
- Perceptron: Rosenblatt, 1958

## Known Issues

- The ordering files (200.txt, 4000.txt) enforce deterministic iteration for reproducibility. Deleting them switches to a seeded random shuffle.
- Matplotlib plots require a GUI backend. On a headless server set matplotlib.use('Agg') first.

## License

See [LICENSE](LICENSE).
