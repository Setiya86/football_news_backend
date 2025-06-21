from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from utils.preprocess import clean_and_prepare_data
import requests

app = Flask(__name__, template_folder='templates')  # Specify the folder for HTML templates
CORS(app)

# Load and process data once at startup
df = pd.read_csv('data/artikel_bola_merged_data.csv')
df_cleaned = clean_and_prepare_data(df).reset_index(drop=True)  # Ensure indices are sequential
df_cleaned['Sumber'] = df_cleaned['Sumber'].fillna('Unknown Source')
df_cleaned['Tanggal'] = df_cleaned['Tanggal'].fillna('Unknown Date')

print(df_cleaned[['Judul', 'Sumber', 'Tanggal']].head())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_cleaned['stemmed_judul'])
tfidf_matrix = X.toarray().T  # Transpose so tfidf_matrix[:, i] = article i
feature_names = vectorizer.get_feature_names_out()

@app.route('/')
def home():
    # Render the HTML page
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()

    if not query:
        return jsonify([])  # Return an empty array if the query is empty

    try:
        # Vectorize the query
        q_vec = vectorizer.transform([query]).toarray().reshape(len(feature_names),)
    except Exception as e:
        return jsonify({"error": f"Failed to vectorize query: {str(e)}"}), 400

    similarities = {}

    # Calculate similarity for titles (stemmed_judul)
    for i in range(tfidf_matrix.shape[1]):
        doc_vec = tfidf_matrix[:, i]
        dot = np.dot(doc_vec, q_vec)
        norm_product = np.linalg.norm(doc_vec) * np.linalg.norm(q_vec)
        sim = dot / norm_product if norm_product != 0 else 0
        similarities[i] = {"title_similarity": sim, "content_similarity": 0}

    # Calculate similarity for article content (clean_artikel)
    for i in range(tfidf_matrix.shape[1]):
        doc_vec = tfidf_matrix[:, i]
        dot = np.dot(doc_vec, q_vec)
        norm_product = np.linalg.norm(doc_vec) * np.linalg.norm(q_vec)
        sim = dot / norm_product if norm_product != 0 else 0
        similarities[i]["content_similarity"] = sim

    # Combine scores with priority for title similarity
    combined_scores = {
        i: (0.7 * similarities[i]["title_similarity"] + 0.3 * similarities[i]["content_similarity"])
        for i in similarities
    }

    # Sort by combined scores
    sim_sorted = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_k = [item for item in sim_sorted if item[1] > 0]  # Include all results with scores > 0

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
                "similarity": float(round(score, 4)),  # Include combined similarity score
                "snippet": df_cleaned.iloc[i]['clean_artikel'][:300] + "..."  # Use preprocessed snippet
            })

    # Debugging: Print the number of results being sent
    print(f"Number of results sent: {len(results)}")

    return jsonify(results)

@app.route('/proxy-image')
def proxy_image():
    image_url = request.args.get('url')
    if not image_url:
        return "Image URL is required", 400

    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        return Response(response.content, content_type=response.headers['Content-Type'])
    else:
        return "Failed to fetch image", response.status_code

if __name__ == '__main__':
    app.run(debug=True)
