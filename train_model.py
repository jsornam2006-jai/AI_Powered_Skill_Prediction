import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset

data = pd.read_csv("dataset/career_prediction.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# Separate input features and target

X = data.drop("Career", axis=1)
y = data["Career"]

print("\nFeatures:")
print(list(X.columns))

print("\nCareer classes:")
print(y.unique())


#  Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create machine learning models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


#  Train and compare models

results = {}

print("\n----- MODEL ACCURACY -----")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"{name}: {accuracy * 100:.2f}%")


#  Select the best model

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print(f"Best Accuracy: {results[best_model_name] * 100:.2f}%")


# Create models folder

os.makedirs("models", exist_ok=True)


#  Save model and feature names

model_data = {
    "model": best_model,
    "features": list(X.columns)
}

joblib.dump(model_data, "models/career_model.pkl")


print("\nModel saved successfully!")
print("Location: models/career_model.pkl")