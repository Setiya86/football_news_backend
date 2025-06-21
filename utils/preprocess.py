import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('indonesian'))
stemmer = PorterStemmer()

def clean_text(text):
    """
    Cleans the input text by removing unwanted characters, phrases, and formatting.
    """
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\bgilabola\s*com\b', '', text, flags=re.IGNORECASE)  # Remove "gilabola com" (case-insensitive, handles extra spaces)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # Remove punctuation
    text = re.sub(r'[0-9]', ' ', text)  # Remove numbers
    text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()  # Remove leading and trailing spaces

def preprocess_column(series):
    """
    Preprocesses a pandas Series by cleaning, tokenizing, removing stopwords, and stemming.
    """
    return [
        " ".join(
            [stemmer.stem(w) for w in word_tokenize(clean_text(doc)) if w not in stop_words]
        )
        for doc in series
    ]

def clean_and_prepare_data(df):
    """
    Cleans and prepares the input DataFrame by applying various preprocessing steps.
    """
    # Remove rows with invalid content
    df = df[df["Isi Artikel"] != "Gagal mengambil isi artikel"]
    df = df[df["Isi Artikel"] != "Content ini berisi video"]
    df = df[df["Isi Artikel"] != "Konten tidak ditemukan"]

    # Remove rows with "Sumber" set to "Tidak ditemukan"
    df = df[df["Sumber"] != "Tidak ditemukan"]

    # Rename columns for consistency
    df.rename(columns={
        "URL Gambar": "Image_URL",
        "URL Artikel": "Article_URL",
        "Isi Artikel": "Article_Content",
    }, inplace=True)

    # Remove rows with "jam yang lalu" or similar formats
    df = df[~df['Tanggal'].str.contains(r'jam yang lalu', na=False, case=False)]

    # Remove day names (e.g., "Minggu") from the Tanggal column
    df['Tanggal'] = df['Tanggal'].str.replace(r'^\w+,', '', regex=True).str.strip()

    # Remove the phrase "gilabola com" from the Article_Content column
    df['Article_Content'] = df['Article_Content'].apply(clean_text)

    # Preprocess the Article_Content and Judul columns
    df['clean_artikel'] = preprocess_column(df['Article_Content'])
    df['clean_judul'] = preprocess_column(df['Judul'])

    # Add stemmed_judul column
    df['stemmed_judul'] = df['clean_judul']

    return df
