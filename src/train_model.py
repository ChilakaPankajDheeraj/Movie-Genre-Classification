
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.preprocessing import clean_text


def train_model():

    # Load dataset
    df = pd.read_csv("data/movies.csv")

    # Clean descriptions
    df["clean_description"] = df["description"].apply(clean_text)

    X = df["clean_description"]
    y = df["genre"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create ML pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ])

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, predictions)

    print("\nMODEL TRAINING COMPLETED")
    print("------------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/genre_classifier.pkl")

    print("\nModel saved successfully!")
    print("Location: models/genre_classifier.pkl")

    return model


if __name__ == "__main__":
    train_model()
