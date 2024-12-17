from flask import Flask, request, render_template
from transformers import pipeline
import os
import PyPDF2

# Initialize Flask app
app = Flask(__name__)

# Initialize Hugging Face QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = " ".join(page.extract_text() for page in reader.pages)
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def query_huggingface(pdf_text, user_query):
    """Use Hugging Face's question-answering pipeline."""
    try:
        response = qa_pipeline({"context": pdf_text, "question": user_query})
        return response['answer']
    except Exception as e:
        return f"Error with Hugging Face pipeline: {e}"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    
    # Save and process the uploaded PDF
    file.save(file.filename)
    pdf_text = extract_text_from_pdf(file.filename)
    os.remove(file.filename)  # Clean up after processing
    
    # Pass extracted text to query.html
    return render_template("query.html", text=pdf_text, query=None, answer=None)

@app.route("/query", methods=["POST"])
def query_pdf():
    pdf_text = request.form["text"]
    user_query = request.form["query"]
    answer = query_huggingface(pdf_text, user_query)
    
    # Render query.html with the answer
    return render_template("query.html", text=pdf_text, query=user_query, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
