
## Author

Name: Chilaka Pankaj Dheeraj
Intern ID: CITS9471
Internship: Machine Learning Internship
Duration (weeks): 12 weeks
Project: Movie Genre Classification

# Movie Genre Classification

## Project Overview

Movie Genre Classification is a machine learning application
that predicts movie genres from plot descriptions using
Natural Language Processing.

## Objectives

- Classify movies based on plot descriptions.
- Apply text preprocessing techniques.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classifier.
- Evaluate model performance.
- Build an interactive Streamlit dashboard.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib
- Pytest

## Machine Learning Algorithms

1. TF-IDF Vectorization
2. Logistic Regression

## Dataset

The project contains 1,600 synthetically generated movie
descriptions across eight genres:

- Action
- Comedy
- Drama
- Horror
- Romance
- Thriller
- Sci-Fi
- Animation

## Installation

```bash
python -m pip install -r requirements.txt
python generate_data.py
python -m src.train_model
```

## Run Application

```bash
python -m streamlit run app.py
```

## Run Tests

```bash
python -m pytest -q
```

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Dataset Limitation

The dataset is synthetic and intended for educational
demonstration. Performance on real-world movie descriptions
may differ significantly.
