from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from utils.preprocess import clean_and_prepare_data

app = Flask(__name__)
CORS(app)

# Load and process data once at startup
df = pd.read_csv('data/artikel_bola_merged_data.csv')
df_cleaned = clean_and_prepare_data(df).reset_index(drop=True)  # pastikan indeks urut
df_cleaned['Sumber'] = df_cleaned['Sumber'].fillna('Unknown Source')
df_cleaned['Tanggal'] = df_cleaned['Tanggal'].fillna('Unknown Date')

print(df_cleaned[['Judul', 'Sumber', 'Tanggal']].head())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_cleaned['stemmed_judul'])
tfidf_matrix = X.toarray().T  # Transpose agar tfidf_matrix[:, i] = artikel ke-i
feature_names = vectorizer.get_feature_names_out()

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()

    if not query:
        return jsonify([])  # jika query kosong

    try:
        q_vec = vectorizer.transform([query]).toarray().reshape(len(feature_names),)
    except Exception as e:
        return jsonify({"error": f"Failed to vectorize query: {str(e)}"}), 400

    similarities = {}
    for i in range(tfidf_matrix.shape[1]):
        doc_vec = tfidf_matrix[:, i]
        dot = np.dot(doc_vec, q_vec)
        norm_product = np.linalg.norm(doc_vec) * np.linalg.norm(q_vec)
        sim = dot / norm_product if norm_product != 0 else 0
        similarities[i] = sim

    # Urutkan dan ambil top 5
    sim_sorted = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_k = [item for item in sim_sorted if item[1] > 0][:5]  # Ambil yang punya skor > 0

    if not top_k:
        return jsonify([])  # Return an empty array if no results

    results = []
    for i, score in top_k:
        if 0 <= i < len(df_cleaned):
            results.append({
                "judul": df_cleaned.iloc[i]['Judul'],  # Use original Judul
                "Sumber": df_cleaned.iloc[i]['Sumber'],  # Use original Sumber
                "Tanggal": df_cleaned.iloc[i]['Tanggal'],  # Use original Tanggal
                "url": df_cleaned.iloc[i]['Article_URL'],  # Use original Article_URL
                "img": df_cleaned.iloc[i]['Image_URL'],  # Use original Image_URL
                "similarity": float(round(score, 4)),  # Include similarity score
                "snippet": df_cleaned.iloc[i]['clean_artikel'][:300] + "..."  # Use preprocessed snippet
            })


    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
