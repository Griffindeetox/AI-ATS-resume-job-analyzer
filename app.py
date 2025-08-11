import streamlit as st
import fitz  # PyMuPDF
import io
import re
import nltk
import spacy
from nltk.corpus import stopwords
from docx import Document
from striprtf.striprtf import rtf_to_text
from bs4 import BeautifulSoup

# ---------------- Page setup (must be first Streamlit command) ----------------
st.set_page_config(
    page_title="AI Resume & Job Analyzer",
    page_icon="📄",
    layout="centered",
)

# ---------------- Sidebar: How to Use + GitHub link + credit ----------------
st.sidebar.header("📌 How to Use")
st.sidebar.markdown("""
1. **Upload your Resume** (PDF, DOCX, RTF, or TXT).
2. **Paste the Job Description (JD)** into the text box.
3. Click **Analyze** to see:
   - ✅ Match score  
   - 🗂 Matched keywords  
   - ❌ Missing keywords  
4. Adjust your resume to improve the score.
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<a href="https://github.com/Griffindeetox/AI-ATS-resume-job-analyzer" target="_blank">💻 <b>View Source Code on GitHub</b></a>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 *Made by Adeyemi O*")

# ---------------- Title & tagline ----------------
st.title("🚀 AI Resume & Job Match Analyzer")
st.markdown(
    "<p style='color: gray; font-size: 16px;'>Upload your resume & paste the job description to get instant match scores, missing skills, and keyword insights.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------- Load NLP tools ----------------
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ---------------- Keyword extractor ----------------
def extract_keywords(text: str):
    TECH_SKILLS = {
        "azure", "terraform", "bicep", "monitoring", "kubernetes", "docker", "ci/cd",
        "azure devops", "linux", "windows", "ansible", "github", "python", "powershell",
        "microsoft 365", "teams", "event grid", "functions", "service bus", "sql",
        "infrastructure", "iac", "guardrails", "policies", "landing zone", "networking",
        "paas", "iaas", "platform as a service", "infrastructure as code", "infrastructure as a service",
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
    return keywords.intersection(TECH_SKILLS)

# ---------------- File text extractors ----------------
def extract_text_from_pdf(file_obj) -> str:
    """Extract text from a PDF (Streamlit UploadedFile)."""
    with fitz.open(stream=file_obj.read(), filetype="pdf") as doc:
        text = []
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def extract_text_from_docx(file_obj) -> str:
    """Extract text from a DOCX."""
    data = file_obj.read()
    bio = io.BytesIO(data)
    doc = Document(bio)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_rtf(file_obj) -> str:
    """Extract text from an RTF."""
    data = file_obj.read()
    try:
        return rtf_to_text(data.decode("utf-8", errors="ignore"))
    except Exception:
        return rtf_to_text(data.decode("latin-1", errors="ignore"))

def extract_text_from_txt(file_obj) -> str:
    """Extract text from a TXT/MD-like file."""
    data = file_obj.read()
    try:
        text = data.decode("utf-8")
    except Exception:
        text = data.decode("latin-1", errors="ignore")
    # light markdown/formatting cleanup
    text = re.sub(r"[#*_>`~\-]{1,}", " ", text)
    return text

def extract_text_from_any(file_obj, filename: str) -> str:
    """Unified extractor for resume files."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_obj)
    if name.endswith(".docx"):
        return extract_text_from_docx(file_obj)
    if name.endswith(".rtf"):
        return extract_text_from_rtf(file_obj)
    # .txt (and simple fallbacks)
    return extract_text_from_txt(file_obj)

# ---------------- Inputs ----------------
resume_file = st.file_uploader(
    "📎 Upload Your Resume (.pdf, .docx, .rtf, .txt)",
    type=["pdf", "docx", "rtf", "txt"]
)

jd_text = st.text_area(
    "🧾 Paste the Job Description here:",
    height=220,
    placeholder="Copy and paste the full job description here..."
)

# ---------------- Main analysis ----------------
if resume_file and jd_text.strip():
    # Extract resume text via unified extractor
    resume_text = extract_text_from_any(resume_file, resume_file.name)

    # Previews
    st.subheader("📄 Resume Preview")
    st.text_area("Your Resume Content", resume_text, height=200)

    st.subheader("🧾 Job Description Preview")
    st.text_area("Job Description Content", jd_text, height=200)

    # Keyword matching (simple baseline)
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)

    matched_keywords = resume_keywords.intersection(jd_keywords)
    missing_keywords = jd_keywords.difference(resume_keywords)

    match_score = round(
        (len(matched_keywords) / len(jd_keywords) * 100) if len(jd_keywords) else 0.0, 2
    )

    # Results
    st.subheader("🔍 Resume vs JD Keyword Match")
    st.markdown(f"**Match Score:** {match_score}%")
    st.markdown(f"**Matched Keywords ({len(matched_keywords)}):** `{', '.join(sorted(matched_keywords))}`")
    st.markdown(f"**Missing Keywords ({len(missing_keywords)}):** `{', '.join(sorted(missing_keywords))}`")

    # Suggestions
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
    st.info("Please upload a resume and paste the job description to start the analysis.")

# ---------------- Main page footer (theme-aware) ----------------
st.markdown(
    """
    <style>
      .app-footer {
        text-align: center;
        color: var(--text-color);
        opacity: 0.7;
        margin-top: 1.5rem;
      }
    </style>
    <div class="app-footer">👨‍💻 Made by <b>Adeyemi O</b></div>
    """,
    unsafe_allow_html=True
)

