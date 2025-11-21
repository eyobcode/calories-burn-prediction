from src.data_loader import DataLoader
from src.preprocess import Preprocessor
from src.trainer import Trainer
from src.utils import train_test_split, save_model

def main():
    print("--- FULL PIPELINE: loading real data ---")

    # 1 Load data
    loader = DataLoader("data/exercise.csv", "data/calories.csv")
    df = loader.load_data()

    # 2 Preprocess
    prep = Preprocessor(df)
    prep.encode_gender()

    # Polynomial features (degree 3)
    feature_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    prep.add_polynomial_features(feature_cols, degree=3)

    # Scale all features including polynomial ones
    scaled_cols = ['Gender'] + feature_cols + [f"{col}_2" for col in feature_cols] + [f"{col}_3" for col in feature_cols]
    prep.scale_features(scaled_cols)

    # Split X and y
    X, y = prep.split_x_y('Calories', scaled_cols)

    # 3 Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)

    # 4 Train model
    trainer = Trainer(X_train, y_train)
    trainer.train(n_iters=5000, lr=0.0001)

    # 5 Evaluate
    mse = trainer.evaluate(X_test, y_test)
    print(f"Test MSE: {mse:.4f}")

    # 6 Save model
    save_model(trainer.model, "model/linear_poly_regression.pkl")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
