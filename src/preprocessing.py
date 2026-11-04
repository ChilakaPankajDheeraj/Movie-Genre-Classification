
import re


def clean_text(text):
    """
    Clean movie description text before TF-IDF processing.
    """

    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
