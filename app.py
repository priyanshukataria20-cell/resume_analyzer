import streamlit as st
import joblib
import pypdf
import os
from scoring import rank_candidates

st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("📄 AI-Based Resume Screening & Ranking System")
st.write("Upload candidate PDF resumes and compare them against a Job Description.")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("resume_model.joblib")

model = load_model()

# User Inputs
jd_text = st.text_area("Paste Job Description (JD) Here:", height=150)
uploaded_files = st.file_uploader("Upload Resumes (PDF format only):", type=["pdf"], accept_multiple_files=True)

if st.button("Rank Candidates"):
    if not jd_text.strip():
        st.warning("Please enter a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
    else:
        # Extract text from PDFs
        pdf_resumes = {}
        for file in uploaded_files:
            try:
                reader = pypdf.PdfReader(file)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
                if len(text.strip()) > 50:
                    pdf_resumes[file.name] = text
            except Exception as e:
                st.error(f"Could not read {file.name}: {e}")

        if pdf_resumes:
            # Rank Candidates
            results, predicted_category, skills_needed = rank_candidates(model, jd_text, pdf_resumes)
            
            st.success(f"**Predicted Job Category:** {predicted_category}")
            st.write(f"**Extracted Job Skills:** {', '.join(sorted(skills_needed)) if skills_needed else 'None detected'}")
            
            st.subheader("Candidate Rankings")
            st.dataframe(results, use_container_width=True)
        else:
            st.error("No readable text found in the uploaded PDFs.")
