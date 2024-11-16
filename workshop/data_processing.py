import nltk
import numpy as np
import os
import pandas as pd
import pickle
import re

from datasets import Dataset
from deep_translator import GoogleTranslator
from langdetect import detect
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

url_pattern = re.compile(r'http\S+|www\S+|https\S+')
special_char_pattern = re.compile(r'[^\w\s]')
number_pattern = re.compile(r'\d+')

def preprocess_reviews(df, 
                       mask_links=True, 
                       remove_numbers=True, 
                       remove_special_chars=True, 
                       min_length=5, 
                       max_length=500):
    """
    Preprocess text data with options for masking links, removing numbers, special characters,
    and filtering reviews by minimum and maximum length

    Parameters:
    - df: DataFrame containing the review data
    - mask_links: whether to replace URLs with 'URL'
    - remove_numbers: whether to remove numeric characters
    - remove_special_chars: whether to remove special characters
    - min_length: minimum allowed word count for reviews
    - max_length: maximum allowed word count for reviews

    Returns:
    - DataFrame with processed reviews
    """

    df = df.copy()

    # compile regex patterns
    url_pattern = re.compile(r'http\S+|www\S+|https\S+')
    special_char_pattern = re.compile(r'[^\w\s]')
    number_pattern = re.compile(r'\d+')
    stop_words = set(stopwords.words('english'))

    if mask_links:
        df.loc[:, 'review'] = df['review'].str.replace(url_pattern, 'URL', regex=True)
    
    if remove_special_chars:
        df.loc[:, 'review'] = df['review'].str.replace(special_char_pattern, ' ', regex=True)

    if remove_numbers:
        df.loc[:, 'review'] = df['review'].str.replace(number_pattern, '', regex=True)
    
    df.loc[:, 'review'] = df['review'].str.lower()
    
    # remove stopwords
    df.loc[:, 'processed_review'] = df['review'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )
    
    # filter reviews by word count
    df = df[df['processed_review'].apply(lambda x: min_length <= len(x.split()) <= max_length)]
    
    return df
    

def create_sampled_reviews(df_reviews_cleaned, total_sample_size=5000, random_state=42):
    """
    Samples reviews from a DataFrame, ensuring 80% are English reviews 
    and 20% are non-English reviews, with balanced `is_fake` values.
    
    Parameters:
        df_reviews_cleaned (pd.DataFrame): Input DataFrame containing review data.
        total_sample_size (int): Total number of samples to return.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: A shuffled, sampled DataFrame of reviews.
    """
    english_sample_size = int(total_sample_size * 0.8)
    remaining_sample_size = total_sample_size - english_sample_size

    df_english = pd.concat([
        df_reviews_cleaned[
            (df_reviews_cleaned['language'] == 'en') & (df_reviews_cleaned['is_fake'] == 0)
        ].sample(english_sample_size // 2, random_state=random_state),
        df_reviews_cleaned[
            (df_reviews_cleaned['language'] == 'en') & (df_reviews_cleaned['is_fake'] == 1)
        ].sample(english_sample_size // 2, random_state=random_state)
    ], ignore_index=True)

    non_english_languages = df_reviews_cleaned['language'].unique()
    non_english_languages = [lang for lang in non_english_languages if lang != 'en']

    df_non_english = pd.DataFrame()
    for lang in non_english_languages:
        lang_df = df_reviews_cleaned[df_reviews_cleaned['language'] == lang]
        lang_sample_size = min(len(lang_df), remaining_sample_size // len(non_english_languages)) // 2

        is_fake_0 = lang_df[lang_df['is_fake'] == 0].sample(
            min(len(lang_df[lang_df['is_fake'] == 0]), lang_sample_size),
            random_state=random_state
        )
        is_fake_1 = lang_df[lang_df['is_fake'] == 1].sample(
            min(len(lang_df[lang_df['is_fake'] == 1]), lang_sample_size),
            random_state=random_state
        )

        df_non_english = pd.concat([df_non_english, is_fake_0, is_fake_1], ignore_index=True)

    df_reviews_small = pd.concat([df_english, df_non_english], ignore_index=True)
    df_reviews_small = df_reviews_small.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_reviews_small


def prepare_data_for_training(df, text_column_name, label_column_name, tokenizer, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    def preprocess_function(examples):
        return tokenizer(examples[text_column_name], truncation=True)
    
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    tokenized_train = tokenized_train.rename_column(label_column_name, "labels")
    tokenized_test = tokenized_test.rename_column(label_column_name, "labels")

    return tokenized_train, tokenized_test

def process_multilingual_reviews(
    df_reviews_cleaned, text_column_name, label_column_name, model_name, fine_tuned_trainer
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    languages = df_reviews_cleaned['language'].unique()
    language_metrics = {}
    sampled_reviews = {}

    for lang in languages:
        lang_df = df_reviews_cleaned[df_reviews_cleaned['language'] == lang]

        if len(lang_df) == 0:
            print(f"Skipping {lang}: No data available.")
            continue

        dataset = Dataset.from_pandas(lang_df)

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        predictions = fine_tuned_trainer.predict(tokenized_dataset)
        preds = predictions.predictions.argmax(-1)
        labels = predictions.label_ids

        if len(labels) == 0 or len(preds) == 0:
            print(f"Skipping {lang}: No true or predicted samples.")
            continue

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        language_metrics[lang] = {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1 Score": f1}

        real_reviews = lang_df[lang_df[label_column_name] == 0]
        fake_reviews = lang_df[lang_df[label_column_name] == 1]

        if len(real_reviews) >= 5 and len(fake_reviews) >= 5:
            real_samples = real_reviews.sample(5, random_state=42)
            fake_samples = fake_reviews.sample(5, random_state=42)
            sampled_reviews[lang] = pd.concat([real_samples, fake_samples])
        else:
            print(f"Skipping {lang}: Not enough samples available.")

    for lang, reviews in sampled_reviews.items():
        reviews['translated_review'] = reviews[text_column_name].apply(
            lambda x: GoogleTranslator(source=lang, target="en").translate(x)
        )

    if sampled_reviews:
        sampled_reviews_combined = pd.concat(sampled_reviews.values(), ignore_index=True)
    else:
        raise ValueError("No valid sampled reviews available for concatenation.")

    metrics_df = pd.DataFrame(language_metrics).T.reset_index().rename(columns={"index": "Language"})
    metrics_df = metrics_df.sort_values(by="F1 Score", ascending=False)

    return metrics_df, sampled_reviews_combined

def evaluate_model_by_language(trainer, df, text_column_name, label_column_name, language_column_name, tokenizer):
    """
    Evaluate the fine-tuned model on each language in the dataset.

    Parameters:
    - trainer: Fine-tuned Hugging Face Trainer.
    - df: DataFrame containing the data.
    - text_column_name: Column name for text input.
    - label_column_name: Column name for labels.
    - language_column_name: Column name for language.
    - tokenizer: Hugging Face tokenizer.

    Returns:
    - pd.DataFrame: A DataFrame containing evaluation metrics for each language.
    """
    languages = df[language_column_name].unique()
    results = []

    for lang in languages:
        print(f"Evaluating for language: {lang}")
        
        lang_df = df[df[language_column_name] == lang]

        if lang_df.empty:
            print(f"No data available for language: {lang}")
            continue
        
        lang_dataset = Dataset.from_pandas(lang_df)

        def preprocess_function(examples):
            return tokenizer(
                examples[text_column_name], truncation=True, padding="max_length"
            )

        tokenized_lang_dataset = lang_dataset.map(preprocess_function, batched=True)

        if label_column_name in tokenized_lang_dataset.column_names:
            tokenized_lang_dataset = tokenized_lang_dataset.rename_column(label_column_name, "labels")

        required_columns = ["input_ids", "attention_mask", "labels"]
        if not all(col in tokenized_lang_dataset.column_names for col in required_columns):
            print(f"Skipping language {lang}: Missing required columns in tokenized dataset.")
            continue

        try:
            predictions = trainer.predict(tokenized_lang_dataset)
            logits = predictions.predictions
            labels = predictions.label_ids
            preds = logits.argmax(axis=-1)

            acc = accuracy_score(labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary", zero_division=0
            )
            
            results.append({
                "Language": lang,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })
        except Exception as e:
            print(f"Error during evaluation for language {lang}: {e}")
            continue

    results_df = pd.DataFrame(results)
    return results_df


def compute_metrics(eval_pred):
    """
    Compute classification metrics (Accuracy, Precision, Recall, F1-score).

    Parameters:
    - eval_pred: Tuple of logits and labels from the model.

    Returns:
    - dict: Computed evaluation metrics.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_fine_tuned_model(trainer, eval_results, output_dir="./fine_tuned_model"):
    """
    Save the fine-tuned model, tokenizer, and evaluation results.

    Parameters:
    - trainer: The Trainer object after fine-tuning.
    - eval_results: Dictionary of evaluation results.
    - output_dir: Directory to save the model and results.
    """
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "eval_results.pkl"), "wb") as f:
        pickle.dump(eval_results, f)

    print(f"Model and results saved to {output_dir}")

def load_fine_tuned_model(output_dir="./fine_tuned_model"):
    """
    Load the fine-tuned model, tokenizer, and evaluation results.

    Parameters:
    - output_dir: Directory where the model and results are saved.

    Returns:
    - model: The loaded fine-tuned model.
    - tokenizer: The loaded tokenizer.
    - eval_results: Dictionary of evaluation results.
    """
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    with open(os.path.join(output_dir, "eval_results.pkl"), "rb") as f:
        eval_results = pickle.load(f)

    print(f"Model and results loaded from {output_dir}")
    return model, tokenizer, eval_results


def evaluate_model_by_language(model, tokenizer, df, text_column_name, label_column_name, language_column_name):
    """
    Evaluate the fine-tuned model on each language in the dataset.

    Parameters:
    - model: Fine-tuned Hugging Face model.
    - tokenizer: Hugging Face tokenizer.
    - df: DataFrame containing the data.
    - text_column_name: Column name for text input.
    - label_column_name: Column name for labels.
    - language_column_name: Column name for language.

    Returns:
    - pd.DataFrame: A DataFrame containing evaluation metrics for each language.
    """
    # define minimal Trainer arguments for evaluation
    training_args = TrainingArguments(
        output_dir="./evaluation_results",
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        do_train=False,
    )

    # create a Trainer instance with the loaded model and tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )

    # evaluate by language
    languages = df[language_column_name].unique()
    results = []

    for lang in languages:
        print(f"Evaluating for language: {lang}")
        
        lang_df = df[df[language_column_name] == lang]

        if lang_df.empty:
            print(f"No data available for language: {lang}")
            continue
        
        lang_dataset = Dataset.from_pandas(lang_df)

        def preprocess_function(examples):
            return tokenizer(
                examples[text_column_name], truncation=True, padding="max_length"
            )

        tokenized_lang_dataset = lang_dataset.map(preprocess_function, batched=True)

        if label_column_name in tokenized_lang_dataset.column_names:
            tokenized_lang_dataset = tokenized_lang_dataset.rename_column(label_column_name, "labels")

        required_columns = ["input_ids", "attention_mask", "labels"]
        if not all(col in tokenized_lang_dataset.column_names for col in required_columns):
            print(f"Skipping language {lang}: Missing required columns in tokenized dataset.")
            continue

        try:
            predictions = trainer.predict(tokenized_lang_dataset)
            logits = predictions.predictions
            labels = predictions.label_ids
            preds = logits.argmax(axis=-1)

            acc = accuracy_score(labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary", zero_division=0
            )
            
            results.append({
                "Language": lang,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })
        except Exception as e:
            print(f"Error during evaluation for language {lang}: {e}")
            continue

    results_df = pd.DataFrame(results)
    return results_df

def load_data():
    df_reviews = pd.read_feather("/home/sagemaker-user/pyladies_workshop_booking/data/reviews_sample.feather")
    return df_reviews


