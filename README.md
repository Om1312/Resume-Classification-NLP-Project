# Resume-Classification-NLP-Project

Resume Classification NLP Project

Automatically classify resumes into job categories (e.g., Data Scientist, HR, Developer, etc.) using Machine Learning + NLP and deploy it using Streamlit.

🚀 Project Overview

This project builds a Resume Category Classifier using:

Natural Language Processing (NLP)

TF-IDF Vectorization

SVM (Support Vector Machine)

OneVsRestClassifier

Streamlit UI

PDF/DOCX/TXT resume upload

The model predicts the category of resumes based on content using text-processing and machine-learning techniques.

🧠 How It Works
1. Cleaning the Resume Text

Remove URLs

Remove special characters

Remove emojis

Convert to lowercase

Remove repeated whitespace

2. Vectorization (TF-IDF)

We convert text → numerical form using TF-IDF, which gives weighted importance to words.

3. Model Training

TF-IDF vectors converted to dense arrays

SVC trained using OneVsRestClassifier

Labels encoded using LabelEncoder

4. Saving Components

We save:

clf.pkl → Trained SVM model

tfidf.pkl → TF-IDF Vectorizer

encoder.pkl → Label Encoder

5. Streamlit App

The user uploads a resume → text extracted → cleaned → vectorized → category predicted.

📁 Project Structure
resume_classifier/
│── app.py
│── clf.pkl
│── encoder.pkl
│── tfidf.pkl
│── requirements.txt
│── README.md
│── data/
│    └── resumes.csv
│── notebooks/
│    └── modelbuilding.ipynb

🔧 Installation
1. Clone the project
git clone <your-repo-link>
cd resume_classifier

2. Install dependencies
pip install -r requirements.txt

🧪 Run the App
streamlit run app.py


Your browser will open automatically.

📤 Upload Resume

You can upload:

.pdf

.docx

.txt

The app extracts text automatically and predicts job category.

🛠 Technologies Used
Area	Tools
NLP	TF-IDF, regex cleaning
ML	SVC, OneVsRestClassifier
Deployment	Streamlit
File Extraction	PyPDF2, docx2txt
Preprocessing	LabelEncoder, NLTK
💡 Key Features

Upload-only resume input

Supports PDF, DOCX, TXT

Clean & preprocess text automatically

Predicts job role instantly

Professional UI with Streamlit

Uses dense TF-IDF vectors (matching training pipeline)
