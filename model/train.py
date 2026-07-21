import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("Titanic Survival Prediction - Model Training")
print("=" * 60)

# -------------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------------
df = pd.read_csv("data/titanic.csv")

print("\nDataset Loaded Successfully!")
print("-" * 60)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["Survived"].value_counts())

# -------------------------------------------------------------------
# Feature Selection
# -------------------------------------------------------------------
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

target = "Survived"

X = df[features]
y = df[target]

# -------------------------------------------------------------------
# Numerical & Categorical Columns
# -------------------------------------------------------------------
numeric_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch"
]

categorical_features = [
    "Pclass",
    "Sex",
    "Embarked"
]

# -------------------------------------------------------------------
# Preprocessing Pipelines
# -------------------------------------------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -------------------------------------------------------------------
# Build Pipeline
# -------------------------------------------------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# -------------------------------------------------------------------
# Train/Test Split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# -------------------------------------------------------------------
# Train Model
# -------------------------------------------------------------------
print("\nTraining Random Forest Model...")

model.fit(X_train, y_train)

print("Training Completed Successfully!")

# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# -------------------------------------------------------------------
# Save Model
# -------------------------------------------------------------------
joblib.dump(model, "model/titanic_model.pkl")

print("\nModel saved successfully!")
print("Location: model/titanic_model.pkl")

print("\n" + "=" * 60)
print("Training Completed Successfully!")
print("=" * 60)
