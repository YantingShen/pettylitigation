import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from transformers import pipeline
import fitz  # PyMuPDF

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Initialize Hugging Face pipelines
relevance_pipeline = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')
analysis_pipeline = pipeline('text-generation', model='gpt2')

@app.route('/analyze-documents', methods=['POST'])
def analyze_documents():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    file_contents = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        if filename.lower().endswith('.pdf'):
            file_contents.append(extract_text_from_pdf(file_path))
        else:
            with open(file_path, 'r') as f:
                file_contents.append(f.read())

    try:
        relevance_results = check_relevance(file_contents)
        if not relevance_results['is_relevant']:
            return jsonify({'error': 'The uploaded documents are not relevant to the issue described.'}), 400

        analysis_results = analyze_documents_content(file_contents)
        return jsonify({'violations': analysis_results})
    except Exception as e:
        print(f'Error analyzing documents: {e}')
        return jsonify({'error': 'Failed to analyze documents'}), 500

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def check_relevance(file_contents):
    prompt = (
        "You are a legal expert. Determine if the following documents are relevant to a legal issue involving "
        "property, employment, contracts, or leasehold agreements. Respond with 'Relevant' or 'Not Relevant' for each document.\n\n"
        "Documents:\n" + "\n\n".join(file_contents)
    )

    response = relevance_pipeline(prompt)
    relevance_text = response[0]['label']
    is_relevant = 'relevant' in relevance_text.lower()
    return {'is_relevant': is_relevant}

def analyze_documents_content(file_contents):
    prompt = (
        "You are a legal expert. Analyze the following documents and identify any violations of terms or clauses. "
        "Provide a detailed description of each violation.\n\n"
        "Documents:\n" + "\n\n".join(file_contents)
    )

    response = analysis_pipeline(prompt, max_length=1500)
    analysis_text = response[0]['generated_text'].strip()
    return parse_violations(analysis_text)

def parse_violations(text):
    violations = []
    lines = text.split('\n')
    current_violation = {}

    for line in lines:
        if line.startswith('Clause:'):
            if current_violation:
                violations.append(current_violation)
                current_violation = {}
            current_violation['clause'] = line.replace('Clause:', '').strip()
        elif line.startswith('Description:'):
            current_violation['description'] = line.replace('Description:', '').strip()

    if current_violation:
        violations.append(current_violation)

    return violations

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(port=3000, debug=True)