class Preprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.X = None
        self.y = None
        self.scaler = {}  # store mean/std for each column

    def encode_gender(self, col='Gender'):
        # robust cleaning: strip + lower then map to 0/1
        # uses pandas string ops for convenience in EDA, but values are stored as floats
        self.df[col] = self.df[col].astype(str).str.strip().str.lower()
        mapped = []
        for val in self.df[col].tolist():
            if val == 'male':
                mapped.append(0.0)
            elif val == 'female':
                mapped.append(1.0)
            else:
                mapped.append(0.0)
        self.df[col] = mapped

    def add_polynomial_features(self, feature_cols, degree=2):
        for col in feature_cols:
            base_values = [float(v) for v in self.df[col].tolist()]
            if degree >= 2:
                self.df[f"{col}_2"] = [x * x for x in base_values]
            if degree >= 3:
                self.df[f"{col}_3"] = [x * x * x for x in base_values]

    def scale_features(self, feature_cols):
        for col in feature_cols:
            values = [float(v) for v in self.df[col].tolist()]
            n = len(values)
            mean = sum(values) / n
            variance = sum((x - mean) ** 2 for x in values) / n
            std = variance ** 0.5
            if std == 0:
                std = 1.0
            self.scaler[col] = (mean, std)
            scaled = [(x - mean) / std for x in values]
            self.df[col] = scaled

    def split_x_y(self, target_col, feature_cols):
        X = []
        y = []
        n = len(self.df)
        for i in range(n):
            row_i = self.df.iloc[i]
            row = [float(row_i[col]) for col in feature_cols]
            X.append(row)
            y.append(float(row_i[target_col]))
        self.X = X
        self.y = y
        return X, y
