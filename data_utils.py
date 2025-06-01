import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


def load_data(parquet_path: str) -> pd.DataFrame:
    """
    Load the SMS spam dataset from a Parquet file and filter to the
    'ham_True_False' template. Drop unnecessary columns.
    """
    df = pd.read_parquet(parquet_path)
    df = df[df['template_name'] == 'ham_True_False'].copy()
    df.drop(columns=['template_name', 'template', 'rendered_input', 'rendered_output'], inplace=True)
    return df


def preprocess_text(text: str) -> str:
    """
    Lowercase → tokenize → remove non-alphanumeric → remove stopwords → stem → join.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns for Length_of_SMS, words_in_sms, sentences_in_sms.
    """
    df['Length_of_SMS'] = df['sms'].apply(len)
    df['words_in_sms'] = df['sms'].apply(lambda x: len(word_tokenize(x)))
    df['sentences_in_sms'] = df['sms'].apply(lambda x: len(nltk.sent_tokenize(x)))
    return df
