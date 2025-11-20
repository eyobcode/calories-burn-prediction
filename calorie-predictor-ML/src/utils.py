import pickle
import random


def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    total = 0.0
    for i in range(n):
        diff = y_true[i] - y_pred[i]
        total += diff * diff
    return total / n


def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)



def train_test_split(X, y, test_size=0.2, random_seed=None):

    # 1. Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)

    # 2. Combine X and y to shuffle together
    combined = list(zip(X, y))
    print("Combined before shuffle:", combined)  # Example output
    random.shuffle(combined)
    print("Combined after shuffle:", combined)

    # 3. Separate X and y again
    X, y = zip(*combined)
    X = list(X)
    y = list(y)
    print("X after unzip:", X)
    print("y after unzip:", y)

    # 4. Compute number of test samples
    n_total = len(X)
    n_test = int(n_total * test_size)
    print("Total samples:", n_total, "Test samples:", n_test)

    # 5. Split into train and test
    X_train = X[:-n_test]  # Everything except last n_test
    X_test = X[-n_test:]   # Last n_test samples
    y_train = y[:-n_test]
    y_test = y[-n_test:]

    print("X_train:", X_train)
    print("y_train:", y_train)
    print("X_test:", X_test)
    print("y_test:", y_test)

    return X_train, X_test, y_train, y_test
