
import pandas as pd

from src.preprocessing import clean_text


def test_clean_text_lowercase():
    result = clean_text("ACTION MOVIE")
    assert result == "action movie"


def test_clean_text_special_characters():
    result = clean_text("Hello! World@123")
    assert result == "hello world"


def test_clean_text_extra_spaces():
    result = clean_text("  hello    world  ")
    assert result == "hello world"


def test_model_prediction():
    import joblib

    model = joblib.load("models/genre_classifier.pkl")

    description = (
        "A scientist discovers a new planet "
        "and investigates alien life forms."
    )

    prediction = model.predict([clean_text(description)])

    assert prediction[0] in model.classes_


def test_dataset():
    df = pd.read_csv("data/movies.csv")

    assert len(df) == 1600
    assert df["genre"].nunique() == 8
    assert df.isnull().sum().sum() == 0
