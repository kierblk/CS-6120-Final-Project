import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap

from matplotlib.patches import Patch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from tabulate import tabulate

import nltk

nltk.download("punkt")


class SentimentAnalyzer:
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        """
        Initialize the sentiment analysis model with the specified model name.

        :param model_name: The name of the pre-trained model to use for sentiment analysis. Defaults
        to "yiyanghkust/finbert-tone".
        :type model_name: str
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline(
            "sentiment-analysis", model=self.model, tokenizer=self.tokenizer
        )

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the given text using the natural language processing model.

        :param text: The text to analyze sentiment for.
        :type text: str
        :return: The sentiment analysis result from the natural language processing model.
        :rtype: object
        """
        return self.nlp(text)

    @staticmethod
    def aggregate_sentiment(df_data):
        """
        Aggregates sentiment data from a DataFrame to calculate and print the overall sentiment based
        on positive, negative, and neutral confidence scores.

        The method processes a DataFrame where each row represents sentiment data. It flattens nested
        sentiment data if necessary, aggregates confidence scores for each sentiment category, calculates
        their proportion in the total confidence, and determines the overall sentiment based on the
        highest proportion.

        :param df_data: DataFrame containing nested lists of dictionaries with keys 'label' and 'score'.
        :type df_data: pandas.DataFrame

        :return: A new DataFrame with the flattened sentiment data and additional columns for aggregated
        results.
        :rtype: pandas.DataFrame

        The method also prints a formatted sentiment summary based on the calculated proportions.
        """
        # Flatten the list of lists into a single list of dictionaries
        flat_data = [item for sublist in df_data for item in sublist]

        # Create a DataFrame from the flattened data
        method_df = pd.DataFrame(flat_data)

        # Step 1: Extract sentiment labels and scores
        sentiment_labels = method_df["label"]
        sentiment_confidences = method_df["score"]

        # Step 2: Aggregate sentiment confidences
        # Calculate the total confidence for each sentiment label
        total_confidence_positive = sum(
            sentiment_confidences[sentiment_labels == "Positive"]
        )
        total_confidence_negative = sum(
            sentiment_confidences[sentiment_labels == "Negative"]
        )
        total_confidence_neutral = sum(
            sentiment_confidences[sentiment_labels == "Neutral"]
        )

        # Step 3: Determine the overall sentiment based on the aggregated confidences
        # Calculate the total confidence across all labels
        total_confidence = (
            total_confidence_positive
            + total_confidence_negative
            + total_confidence_neutral
        )

        # Calculate the proportion of each sentiment label in the overall confidence
        proportion_positive = total_confidence_positive / total_confidence
        proportion_negative = total_confidence_negative / total_confidence
        proportion_neutral = total_confidence_neutral / total_confidence

        # Determine the overall sentiment label based on the highest proportion
        if (
            proportion_positive > proportion_negative
            and proportion_positive > proportion_neutral
        ):
            overall_sentiment = "Positive"
        elif (
            proportion_negative > proportion_positive
            and proportion_negative > proportion_neutral
        ):
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        # Print the overall sentiment summary
        Utilities.format_sentiment_summary(
            overall_sentiment,
            proportion_positive,
            proportion_neutral,
            proportion_negative,
        )

        return method_df


class Summarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """
        Initialize the Summarizer object with the specified model.

        :param model_name: The name of the model to use for summarization. Default is
        "sshleifer/distilbart-cnn-12-6".
        :type model_name: str
        """
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text, max_length=50, min_length=30, do_sample=False):
        """
        Summarizes the given text using the Hugging Face summarizer model.

        :param text: The input text to be summarized.
        :type text: str

        :param max_length: Maximum length of the summary.
        :type max_length: int

        :param min_length: Minimum length of the summary.
        :type min_length: int

        :param do_sample: Whether to use sampling; this affects the randomness of the output.
        :type do_sample: bool

        :return: The summarized text.
        :rtype: str
        """
        return self.summarizer(
            text, max_length=max_length, min_length=min_length, do_sample=do_sample
        )[0]["summary_text"]

    @staticmethod
    def summarize_event(chunks, summary_size):
        """
        Creates a summary for an event by processing chunks of text and summarizing them iteratively.

        :param chunks: List of text chunks to be summarized.
        :type chunks: list of str

        :param summary_size: Target size for the summary.
        :type summary_size: int

        :return: Final summarized text.
        :rtype: str
        """
        print("Processing...")
        current_summary = None
        max_length = summary_size
        for i, chunk in enumerate(chunks):
            print(f"\tchunk {i} of {len(chunks)}")

            to_summarize = []
            if current_summary:
                to_summarize.append(current_summary)

            to_summarize.append(chunk)
            to_summarize = " ".join(to_summarize)

            if len(to_summarize.split()) > max_length:
                current_summary = summarizer.summarize(
                    to_summarize, max_length=max_length
                ).strip()
                while current_summary[-1] not in [".", "!", "?"] and max_length < 750:
                    max_length += 25
                    current_summary = summarizer.summarize(
                        to_summarize, max_length=max_length
                    ).strip()

            else:
                current_summary = to_summarize

        return current_summary


class Utilities:
    @staticmethod
    def format_sentiment_summary(overall_sentiment, positive, negative, neutral):
        """
        Formats the sentiment summary as a string with overall sentiment and confidence proportions.

        :param overall_sentiment: Overall sentiment label.
        :type overall_sentiment: str

        :param positive: Proportion of positive sentiment.
        :type positive: float

        :param negative: Proportion of negative sentiment.
        :type negative: float

        :param neutral: Proportion of neutral sentiment.
        :type neutral: float

        :return: Formatted sentiment summary.
        :rtype: str
        """
        print(
            f"\n\tOverall sentiment: {overall_sentiment.upper()}\n",
            f" ----------------------------------------\n",
            f"\tPositive confidence: {positive:.2f}\n",
            f"\tNeutral confidence: {neutral:.2f}\n",
            f"\tNegative confidence: {negative:.2f}\n",
        )

    @staticmethod
    def truncate_significant_digits(x, n):
        """
        Truncates the number x to n significant digits.

        :param x: Number to truncate.
        :type x: float

        :param n: Number of significant digits.
        :type n: int

        :return: Number truncated to n significant digits.
        :rtype: float
        """
        if x == 0:
            return x
        return np.round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))

    @staticmethod
    def plot_sentiment_over_time(data):
        """
        Plots sentiment scores over time, using different colors for different sentiment types.

        :param data: DataFrame containing sentiment labels and scores.
        :type data: pandas.DataFrame
        """
        # Extract sentiment labels and scores
        sentiment_labels = data["label"]
        sentiment_scores = data["score"]

        # Convert sentiment scores to reflect positive and negative signs
        converted_scores = sentiment_scores.copy()
        converted_scores[sentiment_labels == "Negative"] *= -1

        # Truncate sentiment scores to a certain number of significant digits
        num_significant_digits = 3
        truncated_scores = [
            Utilities.truncate_significant_digits(score, num_significant_digits)
            for score in converted_scores
        ]

        # Convert truncated_scores to a NumPy array
        truncated_scores = np.array(truncated_scores)

        # Create a list of indices to represent the sequence of data points
        indices = range(1, len(sentiment_labels) + 1)

        # Create a color map for sentiment labels
        color_map = {"Positive": "green", "Negative": "red", "Neutral": "blue"}

        # Create the bar chart
        plt.figure(figsize=(10, 6))
        for i, (index, score, label) in enumerate(
            zip(indices, truncated_scores, sentiment_labels)
        ):
            if label == "Neutral":
                # Split neutral sentiment into positive and negative parts
                positive_score = max(score / 2, 0)
                negative_score = min((score / 2) * -1, 0)
                plt.bar(index, positive_score, color="blue", alpha=0.2)
                plt.bar(index, negative_score, color="blue", alpha=0.2)
            else:
                plt.bar(index, score, color=color_map[label], alpha=0.7)

        # Add labels and title
        plt.xlabel("Data Points")
        plt.ylabel("Sentiment Confidence")
        plt.title("Sentiment Analysis Over Time")

        # Creating Patch objects for the legend
        legend_handles = [
            Patch(facecolor="green", label="Positive", alpha=0.7),
            Patch(facecolor="blue", label="Neutral", alpha=0.2),
            Patch(facecolor="red", label="Negative", alpha=0.7),
        ]

        # Customize legend with explicit handle and label mapping
        plt.legend(
            handles=legend_handles,
            loc="lower right",
            title="Sentiment",
            title_fontsize="large",
            facecolor="lightgrey",
            shadow=True,
            fancybox=True,
            edgecolor="black",
            framealpha=1,
            labelcolor=[
                h.get_facecolor() for h in legend_handles
            ],  # Ensure label colors match
        )

        # Customize y-axis tick labels
        plt.gca().yaxis.set_major_formatter(
            lambda x, pos: "{:0.2f}".format(abs(x))
        )  # Display absolute values only

        # Show plot
        # Enable minor ticks
        plt.minorticks_on()
        plt.grid(
            True,
            which="both",
            axis="both",
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_me_table(method_df):
        """
        Displays a formatted table in the console that includes sentiment analysis results and text.

        This function modifies the input DataFrame by adding columns for formatted sentiment scores and wrapped text.
        It then prints a table with these new columns, using a grid format.

        :param method_df: DataFrame containing the data to display. Expected to have columns 'sentiment' and
        'processed_text'.
        :type method_df: pandas.DataFrame

        :raises KeyError: If 'sentiment' or 'processed_text' columns are missing in the DataFrame.

        Example:
            +----------------+--------------------------------------+
            | Sentiment      | Text                                 |
            +================+======================================+
            | Positive: 0.950| This is an example text that will be |
            | Negative: 0.050| wrapped.                             |
            +----------------+--------------------------------------+
        """
        print(f"Sentiment analysis complete!\n")
        method_df["formatted_sentiment"] = method_df["sentiment"].apply(
            lambda s: "\n".join(f"{dct['label']}: {dct['score']:.3f}" for dct in s)
        )
        method_df["wrapped_text"] = method_df["processed_text"].apply(
            Utilities.wrap_long_text
        )
        print(
            tabulate(
                method_df[["formatted_sentiment", "wrapped_text"]],
                headers=["Sentiment", "Text"],
                tablefmt="grid",
            )
        )

    @staticmethod
    def preprocess_text(text):
        """
        Strips and preprocesses the input text for further processing.

        :param text: The text to preprocess.
        :type text: str

        :return: The preprocessed text.
        :rtype: str
        """
        return text.strip()

    @staticmethod
    def chunk_it(text, chunk_size):
        """
        Splits the given text into chunks based on the specified size, ensuring that chunks
        end at sentence boundaries.

        :param text: The text to chunk.
        :type text: str

        :param chunk_size: The approximate size of each chunk in number of words.
        :type chunk_size: int

        :return: A list of text chunks.
        :rtype: list of str
        """
        sentences = nltk.sent_tokenize(text)

        chunks = []
        current_chunk = []
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(" ".join(current_chunk).split()) >= chunk_size:
                chunks.append(" ".join(current_chunk[:-1]))
                current_chunk = [current_chunk[-1]]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    def wrap_long_text(text, width=90):
        """
        Wraps the input text to the specified width.

        :param text: The text to be wrapped.
        :type text: str

        :param width: The maximum width of the wrapped lines, default is 90 characters.
        :type width: int

        :return: The text wrapped to the specified width with newline characters.
        :rtype: str
        """
        return "\n".join(textwrap.wrap(text, width))


if __name__ == "__main__":
    # LOAD FILE
    file = "FOMC-Dataset-SingleEvent-1.csv"
    print(f"Loading {file} ...")
    df = pd.read_csv(file)

    # INSTANTIATE
    sentiment_analyzer = SentimentAnalyzer()
    summarizer = Summarizer()

    # PREPARE DATA
    df_date = df.loc[0]["Text"]
    df_title = Utilities.wrap_long_text(df.loc[1]["Text"], 120)

    filtered_df = df[df["Type"] == 4][["Text"]]
    filtered_df["processed_text"] = filtered_df["Text"].apply(Utilities.preprocess_text)
    all_text = " ".join(filtered_df["processed_text"])  # Concatenate all text elements

    print(f"\nBeginning analysis...\n")

    # SENTIMENT
    filtered_df["sentiment"] = filtered_df["processed_text"].apply(
        sentiment_analyzer.analyze_sentiment
    )
    Utilities.show_me_table(filtered_df)

    print(f"\nChunking transcript for summarization...\n")

    # SUMMARIZATION
    summary_chunks = Utilities.chunk_it(text=all_text, chunk_size=250)
    summary = summarizer.summarize_event(summary_chunks, 200)

    print(f"""
        SUMMARY:
     --------------------
    
{df_date}
{df_title}
    
     
{Utilities.wrap_long_text(summary, 120)}
    """)

    sentiment_data = sentiment_analyzer.aggregate_sentiment(filtered_df["sentiment"])
    Utilities.plot_sentiment_over_time(sentiment_data)
