import streamlit as st
import pickle
import re
import tempfile

import PyPDF2
import docx2txt

# -------------------------------
# Load Saved Models and Vectorizer
# -------------------------------

# Make sure these .pkl files are in the SAME folder as app.py
with open("clf.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


# -------------------------------
# Cleaning function (same as notebook)
# -------------------------------

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText


# -------------------------------
# Extract text from uploaded file
# -------------------------------

def extract_text(uploaded_file):
    # PDF
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    # DOCX
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # docx2txt works best with a real file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
        return text

    # TXT
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    else:
        return None


# -------------------------------
# Prediction function
# -------------------------------

def predict_category(text: str) -> str:
    # 1. Clean text exactly like in notebook
    cleaned_text = cleanResume(text)

    # 2. TF-IDF transform (sparse) → convert to DENSE
    vectorized = tfidf.transform([cleaned_text]).toarray()

    # 3. Predict using loaded SVC model (trained on dense)
    prediction = model.predict(vectorized)

    # 4. Decode label back to original category name
    category_name = encoder.inverse_transform(prediction)
    return category_name[0]


# -------------------------------
# Streamlit UI
# -------------------------------

st.title("📄 Resume Category Classifier")
st.write("Upload your resume (PDF, DOCX, or TXT) and get the predicted job category.")

uploaded_file = st.file_uploader("Upload Resume File", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    st.info(f"File uploaded: **{uploaded_file.name}**")

    if st.button("Predict Category"):
        with st.spinner("Reading and analyzing your resume..."):
            resume_text = extract_text(uploaded_file)

            if not resume_text or resume_text.strip() == "":
                st.error("Could not extract text from the resume. Please try another file.")
            else:
                category = predict_category(resume_text)
                st.success(f"Predicted Category: **{category}**")
else:
    st.warning("Please upload a resume file to continue.")
