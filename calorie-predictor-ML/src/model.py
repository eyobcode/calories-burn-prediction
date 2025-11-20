class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, epsilon=1e-6):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.weights = None
        self.bias = 0

    # Manual dot product for 2 lists
    def dot(self, x_row, weights):
        result = 0
        for x, w in zip(x_row, weights):
            result += x * w
        return result

    # Fit model
    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for iter in range(self.n_iters):
            # Compute predictions
            y_pred = [self.dot(X[i], self.weights) + self.bias for i in range(n_samples)]

            # Compute errors
            errors = [y_pred[i] - y[i] for i in range(n_samples)]

            # Compute gradients manually
            dw = [0.0] * n_features
            for j in range(n_features):
                for i in range(n_samples):
                    dw[j] += (2/n_samples) * errors[i] * X[i][j]
            db = (2/n_samples) * sum(errors)

            # Check convergence: if all weight updates < epsilon
            max_update = max([abs(self.lr * dw_j) for dw_j in dw] + [abs(self.lr * db)])
            if max_update < self.epsilon:
                print(f"Converged at iteration {iter}")
                break

            # Update weights and bias
            for j in range(n_features):
                self.weights[j] -= self.lr * dw[j]
            self.bias -= self.lr * db

    # Predict manually
    def predict(self, x):
        y_pred = [self.dot(x[i], self.weights) + self.bias for i in range(len(x))]
        return y_pred
