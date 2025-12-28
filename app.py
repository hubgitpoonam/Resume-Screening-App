import streamlit as st
import pickle
import re
import nltk
import pdfplumber



# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s*', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)

    return cleanText


# Web App
def main():
    st.title("Resume Screening App")

    upload_file = st.file_uploader(
        "Upload Resume",
        type=["txt", "pdf"]
    )

    if upload_file is not None:
        # --- NEW CODE STARTS HERE ---
        if upload_file.name.endswith('.pdf'):
            with pdfplumber.open(upload_file) as pdf:
                resume_text = ""
                for page in pdf.pages:
                    extract = page.extract_text()
                    if extract:
                        resume_text += extract
        else:
            # Handle .txt files
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode("utf-8")
        # --- NEW CODE ENDS HERE ---

        cleaned_Resume = cleanResume(resume_text)

        # This will now work ONLY if you re-saved tfidf.pkl in your notebook
        input_features = tfidfd.transform([cleaned_Resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate"
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category:", category_name)




# python main

if __name__ == '__main__':
    main()