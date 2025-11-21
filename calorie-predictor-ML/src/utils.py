import pickle
import os
import random


def mean_squared_error(y_true, y_pred, verbose=False):
    n = len(y_true)
    total = 0.0
    for i in range(n):
        diff = float(y_true[i]) - float(y_pred[i])
        total += diff * diff
        if verbose:
            print(f"Actual: {y_true[i]:.3f}, Predicted: {y_pred[i]:.3f}, Error: {diff:.3f}")
    return total / n


def save_model(model, filename):
    os.makedirs("model", exist_ok=True)
    path = os.path.join("model", filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def train_test_split(X, y, test_size=0.2, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    combined = list(zip(X, y))
    random.shuffle(combined)
    Xs, ys = zip(*combined)
    Xs = list(Xs)
    ys = list(ys)

    n_total = len(Xs)
    n_test = int(n_total * test_size)

    if n_test <= 0:
        return Xs, [], ys, []
    if n_test >= n_total:
        return [], Xs, [], ys

    X_train = Xs[:-n_test]
    X_test = Xs[-n_test:]
    y_train = ys[:-n_test]
    y_test = ys[-n_test:]

    return X_train, X_test, y_train, y_test
