import streamlit as st
import fitz  # PyMuPDF
import io
import nltk
import spacy
from nltk.corpus import stopwords

# Page setup
st.set_page_config(page_title="AI Resume & Job Analyzer", layout="centered")
st.title("📄 AI Resume & Job Analyzer for Tech Roles")
st.markdown("Upload your **Resume** and a **Job Description (JD)** to get insights and a compatibility score.")

# Load NLP tools
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# --- Function to extract keywords ---
def extract_keywords(text):
    TECH_SKILLS = {
        "azure", "terraform", "bicep", "monitoring", "kubernetes", "docker", "ci/cd",
        "azure devops", "linux", "windows", "ansible", "github", "python", "powershell",
        "microsoft 365", "teams", "event grid", "functions", "service bus", "sql",
        "infrastructure", "iac", "guardrails", "policies", "landing zone", "networking", "paas", "iaas" "platform as a service", "infrastructure as code", "infrastructure as a service",
    }

    doc = nlp(text.lower())
    keywords = {
        token.lemma_.lower() for token in doc
        if token.is_alpha
        and token.lemma_ not in stop_words
        and len(token.lemma_) > 2
        and token.pos_ in ["NOUN", "PROPN"]
        and token.ent_type_ not in ["DATE", "TIME", "MONEY", "ORDINAL", "CARDINAL"]
    }

    # Return only relevant tech terms
    return keywords.intersection(TECH_SKILLS)

# --- Function to extract text from PDF ---
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# --- File Upload ---
resume_file = st.file_uploader("📎 Upload Your Resume (.txt or .pdf)", type=["txt", "pdf"])
jd_file = st.file_uploader("📎 Upload the Job Description (.txt or .pdf)", type=["txt", "pdf"])

# --- Main Analysis Logic ---
if resume_file and jd_file:

    # Handle resume file
    if resume_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_file)
    else:
        resume_text = resume_file.read().decode("utf-8")

    # Handle JD file
    if jd_file.name.endswith(".pdf"):
        jd_text = extract_text_from_pdf(jd_file)
    else:
        jd_text = jd_file.read().decode("utf-8")

    # Show Previews
    st.subheader("📄 Resume Preview")
    st.text_area("Your Resume Content", resume_text, height=200)

    st.subheader("🧾 Job Description Preview")
    st.text_area("Job Description Content", jd_text, height=200)

    # --- Keyword Matching ---
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)

    matched_keywords = resume_keywords.intersection(jd_keywords)
    missing_keywords = jd_keywords.difference(resume_keywords)

    match_score = round(len(matched_keywords) / len(jd_keywords) * 100, 2)

    # --- Display Results ---
    st.subheader("🔍 Resume vs JD Keyword Match")
    st.markdown(f"**Match Score:** {match_score}%")
    st.markdown(f"**Matched Keywords ({len(matched_keywords)}):** `{', '.join(sorted(matched_keywords))}`")
    st.markdown(f"**Missing Keywords ({len(missing_keywords)}):** `{', '.join(sorted(missing_keywords))}`")

    # --- Smart Suggestions ---
    st.subheader("💡 Copilot-style Suggestions")

    if missing_keywords:
        st.markdown("• Consider adding the following keywords to better align with the JD:")
        st.markdown(f"`{', '.join(sorted(missing_keywords))}`")
    else:
        st.markdown("• ✅ Your resume already covers all the key terms in the job description!")

    if matched_keywords:
        st.markdown("• Your resume already includes:")
        st.markdown(f"`{', '.join(sorted(matched_keywords))}`")

    st.markdown("• Tip: Highlight these skills in your summary or experience section to boost visibility.")

else:
    st.info("Please upload both a resume and a job description to start the analysis.")
