from src.data_loader import DataLoader
from src.preprocess import Preprocessor
from src.trainer import Trainer
from src.utils import train_test_split, save_model

# Small sanity test first (tiny perfect linear data)
from src.model import LinearRegression

print("--- SANITY TEST: tiny linear dataset y=2x ---")
X_small = [[1.0], [2.0], [3.0], [4.0]]
y_small = [2.0, 4.0, 6.0, 8.0]
model_test = LinearRegression(learning_rate=0.1, n_iters=2000, epsilon=1e-8, verbose=True)
model_test.fit(X_small, y_small)
print("learned weights:", model_test.weights, "bias:", model_test.bias)

# Now try full pipeline if data files exist
# import os
# if not (os.path.exists('data/exercise.csv') and os.path.exists('data/calories.csv')):
#     print('\nData files not found in data/. Place exercise.csv and calories.csv there to run the full pipeline.')
# else:
#     print('\n--- FULL PIPELINE: loading real data ---')
#     loader = DataLoader('data/exercise.csv', 'data/calories.csv')
#     df = loader.load_data()
#
#     prep = Preprocessor(df)
#     prep.encode_gender()
#
#     degree = 3
#     feature_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
#     prep.add_polynomial_features(feature_cols, degree=degree)
#
#     # build scaled column list
#     scaled_cols = ['Gender'] + feature_cols
#     if degree >= 2:
#         scaled_cols += [f"{c}_2" for c in feature_cols]
#     if degree >= 3:
#         scaled_cols += [f"{c}_3" for c in feature_cols]
#
#     prep.scale_features(scaled_cols)
#
#     X, y = prep.split_x_y('Calories', scaled_cols)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)
#
#     trainer = Trainer(X_train, y_train, verbose=True)
#     # choose a learning rate; polynomial features may need smaller lr
#     trainer.train(n_iters=5000, lr=0.001)
#
#     mse = trainer.evaluate(X_test, y_test)
#     print("Test MSE:", mse)
#     save_model(trainer.model, "linear_poly_regression.pkl")
#     print("Model saved to model/linear_poly_regression.pkl")
