# 🎭 Sentiment Analysis from Scratch

Binary sentiment classification of product reviews using three linear classifiers implemented entirely from scratch with NumPy — no ML frameworks involved.

## 📖 Description

This project implements and compares three classic linear classification algorithms for sentiment analysis on product reviews:

- **Perceptron** — the foundational online learning algorithm that updates weights whenever a misclassification occurs.
- **Average Perceptron** — a smoothed variant that averages weight vectors across all update steps, reducing sensitivity to the order of training examples.
- **PEGASOS** (Primal Estimated sub-GrAdient SOlver for SVM) — an efficient stochastic sub-gradient descent method for solving the SVM optimization problem with a tunable regularization parameter λ.

Reviews are represented as bag-of-words feature vectors (with stopword removal), and each classifier learns a linear decision boundary to separate positive from negative sentiment.

## 🔬 Methodology

1. **Text preprocessing** — tokenize reviews, strip punctuation/digits, convert to lowercase.
2. **Stopword filtering** — remove common English stopwords loaded from `stopwords.txt`.
3. **Feature extraction** — build a vocabulary from training data and encode each review as a word-count vector.
4. **Training** — fit each classifier (Perceptron, Average Perceptron, PEGASOS) on the training set.
5. **Hyperparameter tuning** — grid search over iterations `T` and regularization `L` using validation accuracy.
6. **Evaluation** — measure classification accuracy on held-out test data.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| 🐍 Language | Python 3 |
| 🔢 Numerics | NumPy |
| 📊 Plotting | Matplotlib |
| 📁 Data format | TSV (tab-separated values) |

## 📦 Dependencies

```
numpy
matplotlib
```

Install with:

```bash
pip install numpy matplotlib
```

## 🚀 How to Run

**Run the test suite** to verify all algorithm implementations:

```bash
python test.py
```

**Run the full pipeline** (train + evaluate on test data):

```bash
python main.py
```

`main.py` loads the review datasets, builds bag-of-words features, trains a PEGASOS classifier with `T=25, L=0.01`, and prints the test accuracy.

To experiment with other classifiers or hyperparameters, uncomment the relevant sections in `main.py` (Problems 5, 7, 8).

## 📂 Project Structure

```
├── project1.py          # Core ML algorithms (perceptron, avg perceptron, PEGASOS)
├── utils.py             # Data loading, plotting, hyperparameter tuning helpers
├── main.py              # Entry point — training and evaluation pipeline
├── test.py              # Unit tests for all algorithm implementations
├── stopwords.txt        # English stopwords list
├── reviews_train.tsv    # Training data
├── reviews_val.tsv      # Validation data
├── reviews_test.tsv     # Test data
├── toy_data.tsv         # 2D toy dataset for visualization
├── 200.txt / 4000.txt   # Fixed shuffle orders for reproducibility
└── B.py                 # Standalone utility (unrelated)
```

## ⚠️ Known Issues

- **Hardcoded file paths** — data files (`reviews_*.tsv`, `stopwords.txt`, shuffle-order files) are loaded by filename only, so scripts must be run from the repository root directory.
- **No CLI arguments** — hyperparameters and dataset paths are set directly in `main.py`; there is no command-line interface.
- **Matplotlib backend** — plotting functions call `plt.show()`, which requires a GUI backend. On headless servers, set `MPLBACKEND=Agg` or comment out plot calls.
- **Feature extraction efficiency** — `extract_bow_feature_vectors` uses `list.count()` inside a loop, resulting in O(n²) per review. For large vocabularies, a `Counter`-based approach would be faster.

## 📚 References

- [PEGASOS: Primal Estimated sub-GrAdient SOlver for SVM (Shalev-Shwartz et al., 2007)](http://www.ee.oulu.fi/research/imag/courses/Vedaldi/ShalevSiSr07.pdf)
- [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain (Rosenblatt, 1958)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.335.3398&rep=rep1&type=pdf)
