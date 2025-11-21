class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, epsilon=1e-6, verbose=False):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.weights = None
        self.bias = 0.0
        self.verbose = verbose
        self.loss_history = []

    def dot(self, x_row, weights):
        result = 0.0
        for x, w in zip(x_row, weights):
            result += x * w
        return result

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for it in range(self.n_iters):
            # predictions
            y_pred = [self.dot(X[i], self.weights) + self.bias for i in range(n_samples)]
            # errors
            errors = [y_pred[i] - y[i] for i in range(n_samples)]

            # compute gradients
            dw = [0.0] * n_features
            for j in range(n_features):
                s = 0.0
                for i in range(n_samples):
                    s += errors[i] * X[i][j]
                dw[j] = (2.0 / n_samples) * s
            db = (2.0 / n_samples) * sum(errors)

            # track loss
            mse = sum((errors[i]) ** 2 for i in range(n_samples)) / n_samples
            self.loss_history.append(mse)
            if self.verbose and (it % 100 == 0 or it == self.n_iters - 1):
                max_up = max(abs(self.lr * d) for d in dw + [db])
                print(f"iter {it}, mse={mse:.6f}, max_update={max_up:.6g}")

            # check convergence
            max_update = max([abs(self.lr * d) for d in dw] + [abs(self.lr * db)])
            if max_update < self.epsilon:
                if self.verbose:
                    print(f"Converged at iteration {it}, mse={mse:.6f}")
                break

            # update
            for j in range(n_features):
                self.weights[j] -= self.lr * dw[j]
            self.bias -= self.lr * db

    def predict(self, X):
        return [self.dot(X[i], self.weights) + self.bias for i in range(len(X))]
