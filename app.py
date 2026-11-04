
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from src.preprocessing import clean_text


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Movie Genre Classification",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Genre Classification System")

st.write(
    "An NLP-based machine learning application "
    "that predicts movie genres from plot descriptions."
)

st.divider()


# -------------------------------------------------
# LOAD DATA AND MODEL
# -------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv("data/movies.csv")


@st.cache_resource
def load_model():
    return joblib.load("models/genre_classifier.pkl")


try:
    df = load_dataset()
    model = load_model()

except FileNotFoundError:
    st.error(
        "Dataset or trained model not found. "
        "Run generate_data.py and train the model first."
    )
    st.stop()


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Movie Genre Prediction",
        "Dataset Explorer",
        "Genre Analytics",
        "Model Performance"
    ]
)

st.sidebar.success("Dataset and model loaded")


# -------------------------------------------------
# PAGE 1: MOVIE GENRE PREDICTION
# -------------------------------------------------

if page == "Movie Genre Prediction":

    st.header("🎯 Predict Movie Genre")

    st.write(
        "Enter a movie plot description to predict "
        "its most likely genre."
    )

    description = st.text_area(
        "Movie Plot Description",
        placeholder=(
            "Example: A scientist discovers a new planet "
            "and investigates mysterious alien life forms..."
        ),
        height=150
    )

    if st.button("Predict Movie Genre", type="primary"):

        if len(description.strip()) < 10:
            st.warning(
                "Please enter a meaningful movie description."
            )

        else:
            cleaned = clean_text(description)

            probabilities = model.predict_proba([cleaned])[0]

            classes = model.classes_

            result = model.predict([cleaned])[0]

            confidence = max(probabilities) * 100

            st.success("Prediction completed!")

            col1, col2 = st.columns(2)

            col1.metric("Predicted Genre", result)

            col2.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.subheader("Genre Probability")

            probability_df = pd.DataFrame({
                "Genre": classes,
                "Probability": probabilities
            }).sort_values(
                "Probability",
                ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Genre")
            )

            st.dataframe(
                probability_df,
                use_container_width=True
            )


# -------------------------------------------------
# PAGE 2: DATASET EXPLORER
# -------------------------------------------------

elif page == "Dataset Explorer":

    st.header("📂 Movie Dataset Explorer")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Movies", len(df))

    col2.metric(
        "Total Genres",
        df["genre"].nunique()
    )

    col3.metric(
        "Unique Movie Titles",
        df["movie_title"].nunique()
    )

    st.subheader("Search Movies")

    search = st.text_input(
        "Search by movie title or description"
    )

    filtered_df = df.copy()

    if search:
        filtered_df = filtered_df[
            filtered_df["movie_title"].str.contains(
                search,
                case=False,
                na=False
            )
            |
            filtered_df["description"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    selected_genre = st.selectbox(
        "Filter by Genre",
        ["All"] + sorted(df["genre"].unique())
    )

    if selected_genre != "All":
        filtered_df = filtered_df[
            filtered_df["genre"] == selected_genre
        ]

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    st.download_button(
        "Download Movie Dataset",
        df.to_csv(index=False),
        file_name="movies.csv",
        mime="text/csv"
    )


# -------------------------------------------------
# PAGE 3: GENRE ANALYTICS
# -------------------------------------------------

elif page == "Genre Analytics":

    st.header("📊 Movie Genre Analytics")

    genre_counts = df["genre"].value_counts()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Genre Distribution")

        st.bar_chart(genre_counts)

    with col2:

        st.subheader("Genre Share")

        fig, ax = plt.subplots()

        ax.pie(
            genre_counts.values,
            labels=genre_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.axis("equal")

        st.pyplot(fig)

    st.subheader("Genre Summary")

    summary = (
        df.groupby("genre")
        .agg(
            Total_Movies=("movie_id", "count")
        )
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True
    )


# -------------------------------------------------
# PAGE 4: MODEL PERFORMANCE
# -------------------------------------------------

elif page == "Model Performance":

    st.header("📈 Model Performance Evaluation")

    X = df["description"].apply(clean_text)
    y = df["genre"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted Genre")
    ax.set_ylabel("Actual Genre")

    st.pyplot(fig)


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Movie Genre Classification | "
    "Machine Learning Internship Project 4"
)
