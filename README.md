# AI Resume Screening App 📄🔍

An end-to-end Machine Learning application that parses resumes (PDF/TXT) and predicts the job category using NLP.

## Features
- **PDF Extraction**: Uses `pdfplumber` to extract text from resumes.
- **NLP Processing**: Cleans text and uses TF-IDF vectorization.
- **Classification**: Predicts from 25+ categories (Java Developer, Data Science, etc.).
- **Interactive UI**: Built with Streamlit for easy file uploads.

## How to Run
1. Clone the repo: `git clone <your-repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
