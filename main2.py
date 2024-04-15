import pandas as pd
import nltk
import string
import textwrap

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from tabulate import tabulate

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")


def show_me_this(thing):
    # Set display options
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.width", 100)
    print(thing)


def show_me_table(df):
    # Apply formatting to each row's 'sentiment' column
    df["formatted_sentiment"] = df["sentiment"].apply(
        lambda s: "\n".join(f"{key}: {value:.3f}" for key, value in s.items())
    )
    # Wrap the text in the 'processed_text' column
    df["wrapped_text"] = df["processed_text"].apply(
        lambda text: "\n".join(textwrap.wrap(text, 50))
    )

    # Use tabulate to print the DataFrame with formatted columns
    print(
        tabulate(
            df[["formatted_sentiment", "wrapped_text"]], headers="keys", tablefmt="grid"
        )
    )


def read_data(file):
    df = pd.read_csv(file)
    return df


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower().strip()

    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)

    return " ".join(filtered_tokens)


def analyze_sentiment_vader(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment


if __name__ == "__main__":
    file = "FOMC-Dataset-SingleEvent-1.csv"
    df = read_data(file)
    # Filter for meeting transcript entries
    # Types:
    # 1 : Date of meeting
    # 2 : Title of meeting
    # 3 : Meeting attendance
    # 4 : Meeting transcripts
    # 5 : Meeting footnotes
    filtered_df = df[df["Type"] == 4][["Text"]]

    # Apply preprocessing to each text entry in the DataFrame
    filtered_df["processed_text"] = filtered_df["Text"].apply(preprocess_text)

    # Sentiment using VADER
    filtered_df["sentiment"] = filtered_df["processed_text"].apply(
        analyze_sentiment_vader
    )

    # Display processed text to verify preprocessing
    show_me_table(filtered_df[["sentiment", "processed_text"]])
