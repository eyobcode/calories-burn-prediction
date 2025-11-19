class Preprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.X = None
        self.y = None

    def encode_gender(self):
        for i in range(len(self.df['Gender'])):
            if self.df['Gender'][i] == 'Male':
                self.df['Gender'][i] = 0
            elif self.df['Gender'][i] == 'Female':
                self.df['Gender'][i] = 1

    def scale_features(self, feature_cols):
        # μ = (1/n) Σ x_i
        # σ² = (1/n) Σ (x_i - μ)²
        # σ = √σ²
        # z_i = (x_i - μ)/σ

        for col in feature_cols:
            # mean
            n = len(self.df[col])
            values = self.df[col].tolist()
            mean = sum(values) / n
            # std
            variance = sum((x - mean)**2 for x in values) / n
            std = variance ** 0.5
            # scale
            for i in range(n):
                self.df[col][i] = (self.df[col][i] - mean) / std

    def split_x_y(self, target_col, feature_cols):
        x = []
        y = []
        for i in range(len(self.df)):
            row = []
            for col in feature_cols:
                row.append(self.df[col][i])
            x.append(row)
            y.append(self.df[target_col][i])
        self.X = x
        self.y = y
        return self.X, self.y
