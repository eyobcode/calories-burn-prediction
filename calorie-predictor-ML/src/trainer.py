from .model import LinearRegression
from .utils import mean_squared_error

class Trainer:
    def __init__(self, x, y):
        self.X = x
        self.y = y
        self.model = LinearRegression()

    def train(self, n_iters=1000, lr=0.01):
        self.model.lr = lr
        self.model.n_iters = n_iters
        self.model.fit(self.X, self.y)

    def evaluate(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse
