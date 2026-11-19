
# ============================================================
# INT395 - Airfare Component Price Estimation
# Step 5: Data Preprocessing
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv("data/Clean_Dataset.csv")

print("Original dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. Remove unnecessary columns
# ------------------------------------------------------------

# Flight number is an identifier, not a generalizable
# numerical or categorical flight characteristic.

df = df.drop(columns=["flight"])

print("Shape after removing flight:", df.shape)

# ------------------------------------------------------------
# 3. Separate features and target
# ------------------------------------------------------------

X = df.drop(columns=["price"])
y = df["price"]

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget column:")
print("price")

# ------------------------------------------------------------
# 4. Identify categorical and numerical features
# ------------------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

# ------------------------------------------------------------
# 5. Create numerical preprocessing pipeline
# ------------------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# ------------------------------------------------------------
# 6. Create categorical preprocessing pipeline
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

# ------------------------------------------------------------
# 7. Combine preprocessing pipelines
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

# ------------------------------------------------------------
# 8. Split dataset into training and testing sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ------------------------------------------------------------
# 9. Fit preprocessing on training data only
# ------------------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")
