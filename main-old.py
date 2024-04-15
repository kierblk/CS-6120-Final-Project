import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from tabulate import tabulate


class SentimentAnalyzer:
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        """
        Initialize the sentiment analysis model with the specified model name.

        :param model_name: str, optional, default is "yiyanghkust/finbert-tone"
            The name of the pre-trained model to use for sentiment analysis.
        :return: None
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the given text using the natural language processing model.

        :param text: The text to analyze sentiment for.
        :type text: str
        :return: The sentiment analysis result from the natural language processing model.
        :rtype: object
        """
        return self.nlp(text)

    def aggregate_sentiment(self, df_data):
        # Flatten the list of lists into a single list of dictionaries
        flat_data = [item for sublist in df_data for item in sublist]

        # Create a DataFrame from the flattened data
        df = pd.DataFrame(flat_data)

        # Step 1: Extract sentiment labels and scores
        sentiment_labels = df['label']
        sentiment_confidences = df['score']

        # Step 2: Aggregate sentiment confidences
        # Calculate the total confidence for each sentiment label
        total_confidence_positive = sum(sentiment_confidences[sentiment_labels == 'Positive'])
        total_confidence_negative = sum(sentiment_confidences[sentiment_labels == 'Negative'])
        total_confidence_neutral = sum(sentiment_confidences[sentiment_labels == 'Neutral'])

        # Step 3: Determine the overall sentiment based on the aggregated confidences
        # Calculate the total confidence across all labels
        total_confidence = total_confidence_positive + total_confidence_negative + total_confidence_neutral

        # Calculate the proportion of each sentiment label in the overall confidence
        proportion_positive = total_confidence_positive / total_confidence
        proportion_negative = total_confidence_negative / total_confidence
        proportion_neutral = total_confidence_neutral / total_confidence

        # Determine the overall sentiment label based on the highest proportion
        if proportion_positive > proportion_negative and proportion_positive > proportion_neutral:
            overall_sentiment = 'Positive'
        elif proportion_negative > proportion_positive and proportion_negative > proportion_neutral:
            overall_sentiment = 'Negative'
        else:
            overall_sentiment = 'Neutral'

        # Print the overall sentiment summary
        summ_overall = f"\nOverall sentiment: {overall_sentiment.upper()}\n"
        summ_pos = f"\tPositive confidence: {proportion_positive:.2f}\n"
        summ_neu = f"\t Neutral confidence: {proportion_neutral:.2f}\n"
        summ_neg = f"\tNegative confidence: {proportion_negative:.2f}\n"

        full_summ = summ_overall + summ_pos + summ_neu + summ_neg

        print(full_summ)

        return df


class Summarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """
        Initialize the Summarizer object with the specified model.

        :param model_name: str, optional
            The name of the model to use for summarization. Default is "sshleifer/distilbart-cnn-12-6".
        """
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text):
        """
        Summarizes the given text using the Hugging Face summarizer model.

        :param text: The input text to be summarized.
        :type text: str
        :return: The summarized text.
        :rtype: str
        """
        return self.summarizer(text, max_length=50, min_length=30, do_sample=False)[0]['summary_text']


class Utilities:
    @staticmethod
    def truncate_significant_digits(x, n):
        """
        Truncates the number x to n significant digits.
        """
        if x == 0:
            return x
        return np.round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))

    @staticmethod
    def plot_sentiment_over_time(sentiment_data):

        # Extract sentiment labels and scores
        sentiment_labels = sentiment_data['label']
        sentiment_scores = sentiment_data['score']

        # Convert sentiment scores to reflect positive and negative signs
        converted_scores = sentiment_scores.copy()
        converted_scores[sentiment_labels == 'Negative'] *= -1

        # Truncate sentiment scores to a certain number of significant digits
        num_significant_digits = 3
        truncated_scores = [Utilities.truncate_significant_digits(score, num_significant_digits) for score in converted_scores]

        # Convert truncated_scores to a NumPy array
        truncated_scores = np.array(truncated_scores)

        # Create a list of indices to represent the sequence of data points
        indices = range(1, len(sentiment_labels) + 1)

        # Create a color map for sentiment labels
        color_map = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}
        colors = [color_map[label] for label in sentiment_labels]

        # Create the bar chart
        plt.figure(figsize=(10, 6))
        handles = ['Positive', 'Neutral', 'Negative']
        for i, (index, score, label) in enumerate(zip(indices, truncated_scores, sentiment_labels)):
            if label == 'Neutral':
                # Split neutral sentiment into positive and negative parts
                positive_score = max(score/2, 0)
                negative_score = min((score/2)*-1, 0)
                plt.bar(index, positive_score, color='blue', alpha=0.2)
                plt.bar(index, negative_score, color='blue', alpha=0.2)
            else:
                plt.bar(index, score, color=color_map[label], alpha=0.7)

        # Add labels and title
        plt.xlabel('Data Points')
        plt.ylabel('Sentiment Confidence')
        plt.title('Sentiment Analysis Over Time')

        # Customize legend
        legend_colors = [color_map[label] for label in handles]
        plt.legend(
            handles,
            handlelength=0,
            loc='lower right',
            labels=handles,
            title='Sentiment',
            title_fontsize='large',
            facecolor='lightgrey',
            shadow=True, fancybox=True,
            edgecolor='black',
            framealpha=1,
            labelcolor=legend_colors
        )

        # Customize y-axis tick labels
        plt.gca().yaxis.set_major_formatter(lambda x, pos: '{:0.2f}'.format(abs(x)))  # Display absolute values only

        # Show plot
        # Enable minor ticks
        plt.minorticks_on()
        plt.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_me_table(df, title, date):
        """
        Displays a formatted table in the console that includes sentiment analysis results and text.

        This function modifies the input DataFrame by adding columns for formatted sentiment scores and wrapped text.
        It then prints a table with these new columns, using a grid format. The table is preceded by a title and date.

        :param df: DataFrame containing the data to display. Expected to have columns 'sentiment' and 'processed_text'.
        :type df: pandas.DataFrame

        :param title: Title of the table to be displayed.
        :type title: str

        :param date: Date associated with the data, displayed above the table.
        :type date: str

        :raises KeyError: If 'sentiment' or 'processed_text' columns are missing in the DataFrame.

        Example:
            +----------------+--------------------------------------+
            | Sentiment      | Text                                 |
            +================+======================================+
            | Positive: 0.950| This is an example text that will be |
            | Negative: 0.050| wrapped.                             |
            +----------------+--------------------------------------+
        """
        print(f"\n{title}\n\n{date}\n")
        df["formatted_sentiment"] = df["sentiment"].apply(
            lambda s: "\n".join(f"{dct['label']}: {dct['score']:.3f}" for dct in s)
        )
        df["wrapped_text"] = df["processed_text"].apply(
            lambda text: "\n".join(textwrap.wrap(text, 90))
        )
        print(
            tabulate(
                df[["formatted_sentiment", "wrapped_text"]], headers=["Sentiment", "Text"], tablefmt="grid"
            )
        )

    @staticmethod
    def preprocess_text(text):
        return text.strip()


if __name__ == "__main__":

    file = "FOMC-Dataset-SingleEvent-1.csv"
    print(f"Loading {file} ...")
    df = pd.read_csv(file)

    df_date = df.loc[0]['Text']
    df_title = "\n".join(textwrap.wrap(df.loc[1]['Text'], 120))

    filtered_df = df[df["Type"] == 4][["Text"]]
    filtered_df["processed_text"] = filtered_df["Text"].apply(Utilities.preprocess_text)

    print(f"\nBeginning sentiment analysis ...\n")
    sentiment_analyzer = SentimentAnalyzer()
    filtered_df["sentiment"] = filtered_df["processed_text"].apply(sentiment_analyzer.analyze_sentiment)
    Utilities.show_me_table(filtered_df, df_title, df_date)
    sentiment_data = sentiment_analyzer.aggregate_sentiment(filtered_df["sentiment"])
    # Call the function with your sentiment data
    Utilities.plot_sentiment_over_time(sentiment_data)

    print(f"\nBeginning summarization ...\n")
    all_text = " ".join(filtered_df["processed_text"])  # Concatenate all text elements

    # print(all_text)

    # Split the text into smaller chunks to avoid input size mismatch
    # chunk_size = 1000
    # chunks = [all_text[i:i + chunk_size] for i in range(0, len(all_text), chunk_size)]
    #
    # sentiment_results = []
    # for chunk in chunks:
    #     sentiment_results.append(sentiment_analyzer.analyze_sentiment(chunk))
    #
    # sentiment_result_all = sum(sentiment_results, [])
    # print(sentiment_result_all)

    # summarizer = Summarizer()
    # summary_text = summarizer.summarize(all_text)
    # print(summary_text)