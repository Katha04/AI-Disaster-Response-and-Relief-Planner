import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load trained model
model = joblib.load("ml/disaster_model.pkl")

# Load preprocessor
preprocessor = joblib.load("ml/preprocessor.pkl")

# Load dataset
df = pd.read_csv("data/disaster_dataset.csv")

# Separate features and actual severity
X = df.drop("severity", axis=1)
y_actual = df["severity"]

# Transform features
X_processed = preprocessor.transform(X)

# Predict severity
predictions = model.predict(X_processed)

# Display predictions
print("Saved model loaded successfully!")
print("\nPredictions:")
print(predictions.tolist())

# Evaluate predictions
accuracy = accuracy_score(y_actual, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_actual, predictions, zero_division=0))