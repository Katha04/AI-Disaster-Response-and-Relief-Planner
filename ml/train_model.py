import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("data/disaster_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. Separate features and target
# ==========================================

X = df.drop("severity", axis=1)
y = df["severity"]


# ==========================================
# 3. Identify columns
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ==========================================
# 4. Split data BEFORE preprocessing
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. Create preprocessor
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# ==========================================
# 6. Create complete ML pipeline
# ==========================================

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(random_state=42)
        )
    ]
)


# ==========================================
# 7. Train model
# ==========================================

model_pipeline.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ==========================================
# 8. Make predictions on test data
# ==========================================

predictions = model_pipeline.predict(X_test)

print("\nActual severities:")
print(y_test.tolist())

print("\nPredicted severities:")
print(predictions.tolist())


# ==========================================
# 9. Calculate test accuracy
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")


# ==========================================
# 10. Classification report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# 11. Confusion matrix
# ==========================================

labels = ["Low", "Medium", "High", "Critical"]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\nConfusion Matrix:")
print("Label order:", labels)
print(cm)


# ==========================================
# 12. Save preprocessor
# ==========================================

joblib.dump(
    model_pipeline.named_steps["preprocessor"],
    "ml/preprocessor.pkl"
)


# ==========================================
# 13. Save trained classifier
# ==========================================

joblib.dump(
    model_pipeline.named_steps["classifier"],
    "ml/disaster_model.pkl"
)


print("\nModel saved successfully!")
print("Preprocessor saved successfully!")
print("\nML training process completed!")