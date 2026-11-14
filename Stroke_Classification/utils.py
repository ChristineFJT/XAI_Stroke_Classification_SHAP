import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing  import OneHotEncoder, StandardScaler

pd.set_option("display.max_columns", None)


class DataLoader:
    NUMERICAL_COLS = ["age", "avg_glucose_level", "bmi"]
    CATEGORICAL_COLS = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]
    BINARY_COLS = ["hypertension", "heart_disease"]
    FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS + BINARY_COLS

    def __init__(self):
        self.data = None


    def load_dataset(self, path="data/healthcare-dataset-stroke-data.csv"):
        self.data = pd.read_csv(path)
        return self.data

    
    def preprocess_data(self):
        
        if self.data is None:
            raise ValueError("Load the dataset before preprocessing it.")

        # Exclude 'id', the target from X
        # Keep 'stroke' separately for stratified splitting
        self.data = self.data[self.FEATURE_COLS + ["stroke"]].copy()
        # Keep 'Unknown' smoking status as it own recorded category
        return self.data

    def get_data_split(self):
        if self.data is None:
            raise ValueError("Load and preprocess the dataset first.")

        X = self.data[self.FEATURE_COLS].copy()
        y = self.data["stroke"].copy()

        # Split dataset
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # Display predictors before splitting
        print("\n=== Predictors (X) ===")
        print("Shape:", X.shape)
        print("Predictor columns:", X.columns.tolist())
        print("\nTrain rows:", len(X_train_raw), "\nTest rows:", len(X_test_raw))

        # Display target before splitting
        print("\n=== Target (y) ===")
        print("Shape:", y.shape)
        print("Target column:", y.name)

        # Display target class distribution
        print("\n=== Target Class Distribution ===")
        print(y.value_counts().sort_index())

        
        print("\nTraining class counts:", y_train.value_counts().sort_index())
        print("\nTesting class counts:", y_test.value_counts().sort_index())

        return X_train_raw, X_test_raw, y_train, y_test

    
    def get_preprocessor(self):
        """Return an unfitted preprocessor; fit only on training data."""
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="constant", fill_value="Missing"
            )),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        return ColumnTransformer(
            [
                ("num", numerical_pipeline, self.NUMERICAL_COLS),
                ("cat", categorical_pipeline, self.CATEGORICAL_COLS),
                ("binary", "passthrough", self.BINARY_COLS),
            ],
            verbose_feature_names_out=False
        )


    def get_model_pipeline(self):
        """Preprocess -> oversample -> train, without touching the test set"""
        return ImbPipeline([
            ("preprocessing", self.get_preprocessor()),
            ("oversampling", RandomOverSampler(random_state=42)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42),),
        ])

    @staticmethod
    def show_oversampling_summary(y_train):
        """Preview class sizes; the actual oversampling happends in model.fit()."""
        counts = y_train.value_counts().sort_index()
        majority_size = int(counts.max())
        print("=== Before Oversampling (Training Only) ===")
        print(counts.to_string())
        print(f"Training total: {len(y_train):,}")

        print("\n=== After Oversampling (Expected Training Sizes) ===")
        for label in counts.index:
            print(f"Class {label}: {majority_size:,}")
        print(f"Training total: {majority_size * len(counts):,}")

    @staticmethod
    def get_transformed_feature_names(fitted_model):
        """Call only after the pipeline has been fitting on training data."""
        return fitted_model.named_steps["preprocessing"].get_feature_names_out()