from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import fitz
import requests
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
import spacy

# Initialize Flask app
app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

# Load models and components
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
scaler = StandardScaler()
pca = PCA(n_components=50, random_state=42)
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
clf = xgb.XGBClassifier()
clf.load_model("models/xgb_model.json")
# save your model to disk first

# Example roles
job_descriptions = {
    "Machine Learning Engineer": "Machine learning, TensorFlow, PyTorch, deep learning, AI models",
    "Data Scientist": "Data analysis, Python, SQL, statistics, machine learning",
    "Web Developer": "HTML, CSS, JavaScript, React, frontend development",
}

# Utility to extract text from uploaded PDF
def extract_text_from_resume(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

# Skill extraction
def extract_skills(text):
    doc = nlp(text.lower())
    return list(set(token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]))

# Match job role
def find_best_match_and_gaps(resume_text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = {}
    missing_skills = {}
    for role, desc in job_descriptions.items():
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_text, desc])
        similarity = cosine_similarity(vectors)[0, 1]
        similarities[role] = similarity

        resume_words = set(resume_text.split())
        job_words = set(desc.lower().split(", "))
        missing_skills[role] = list(job_words - resume_words)

    best_match = max(similarities, key=similarities.get)
    return best_match, missing_skills[best_match]

# Route: Home
@app.route('/')
def home():
    return render_template('index.html')

# Route: Handle file upload
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['resume']
    if not file:
        return "No file uploaded!"

    text = extract_text_from_resume(file)
    skills = extract_skills(text)
    skills_text = " ".join(skills)

    embedding = sbert_model.encode([text])
    num_features = np.array([[22, 8.5]])  # Dummy age and CGPA for now
    num_scaled = scaler.fit_transform(num_features)
    combined = np.hstack((embedding, num_scaled))
    reduced = pca.fit_transform(combined)

    cluster_label = kmeans.predict(reduced)[0]
    job_role = find_best_match_and_gaps(skills_text)
    
    return render_template('index.html', result={
        "cluster_label": cluster_label,
        "suggested_role": job_role[0],
        "missing_skills": job_role[1]
    })

if __name__ == '__main__':
    app.run(debug=True)
