import pandas as pd

class DataLoader:
    def __init__(self, exercise_path="data/exercise.csv", calories_path="data/calories.csv"):
        self.exercise_path = exercise_path
        self.calories_path = calories_path

    def load_data(self):
        exercise = pd.read_csv(self.exercise_path)
        calories = pd.read_csv(self.calories_path)
        df = exercise.merge(calories, on="User_ID", how="inner")
        return df
