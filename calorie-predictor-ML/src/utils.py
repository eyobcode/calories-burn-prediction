import pickle

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
