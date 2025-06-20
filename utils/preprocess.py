import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('indonesian'))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'@\w+', '', text)
    text = text.lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'[0-9]', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def preprocess_column(series):
    return [" ".join(
        [stemmer.stem(w) for w in word_tokenize(clean_text(doc)) if w not in stop_words]
    ) for doc in series]

def clean_and_prepare_data(df):
    df = df[df["Isi Artikel"] != "Gagal mengambil isi artikel"]
    df = df[df["Isi Artikel"] != "Content ini berisi video"]
    df = df[df["Isi Artikel"] != "Konten tidak ditemukan"]

    df.rename(columns={
        "URL Gambar": "Image_URL",
        "URL Artikel": "Article_URL",
        "Isi Artikel": "Article_Content",
    }, inplace=True)

    # Remove rows with "jam yang lalu" or similar formats
    df = df[~df['Tanggal'].str.contains(r'jam yang lalu', na=False, case=False)]

    # Remove day names (e.g., "Minggu") from the Tanggal column
    df['Tanggal'] = df['Tanggal'].str.replace(r'^\w+,', '', regex=True).str.strip()

    # Preprocess the Article_Content and Judul columns
    df['clean_artikel'] = preprocess_column(df['Article_Content'])
    df['clean_judul'] = preprocess_column(df['Judul'])

    # Add stemmed_judul column
    df['stemmed_judul'] = df['clean_judul']

    return df
