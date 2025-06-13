from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from sentence_transformers import SentenceTransformer, util  # Add this import

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model once at startup
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_file(file):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)

    if ext == '.pdf':
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return ' '.join([page.extract_text() or '' for page in reader.pages])
    elif ext == '.docx':
        return docx2txt.process(filepath)
    elif ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

@app.route('/match_resume', methods=['POST'])
def match_resume():
    if 'resume' not in request.files or 'jd' not in request.form:
        return jsonify({'error': 'Missing resume or job description'}), 400

    resume_file = request.files['resume']
    jd_text = request.form['jd']

    resume_text = extract_text_from_file(resume_file)

    documents = [resume_text, jd_text]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(documents)

    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

    # Semantic similarity using SentenceTransformer
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_jd = model.encode(jd_text, convert_to_tensor=True)
    semantic_score = float(util.cos_sim(emb_resume, emb_jd)[0][0]) * 100

    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())
    common = sorted(resume_words.intersection(jd_words), key=lambda x: len(x), reverse=True)[:10]

    return jsonify({
        'tfidf_score': tfidf_score,
        'semantic_score': semantic_score,
        'common_keywords': common
    })

if __name__ == '__main__':
    app.run()
