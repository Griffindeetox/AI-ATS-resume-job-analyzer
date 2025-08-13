import streamlit as st
import fitz  # PyMuPDF
import io
import re
import nltk
import spacy
from nltk.corpus import stopwords
from docx import Document
from striprtf.striprtf import rtf_to_text

# ---------------- Page setup (must be the first Streamlit command) ----------------
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
   - 🗂 Matched keywords/phrases  
   - ❌ Missing keywords/phrases  
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

# ---------------- Dynamic JD-driven keyword/phrase extractor ----------------
# Captures: acronyms (QA, API, ETL, SQL), noun/proper-noun lemmas (skills/tools),
# short noun phrases (e.g., "manual testing", "data integration", "test cases"),
# and selected verbs related to QA/support (testing, troubleshooting, integration).

ALLOWED_VERBS = {
    "test", "testing", "troubleshoot", "troubleshooting", "debug", "debugging",
    "integrate", "integration", "document", "documentation", "support",
    "analyze", "analysis"
}

# Simple synonym/normalization folding
SYNONYMS = {
    "rest apis": "rest api",
    "http status codes": "http status",
    "qa testing": "qa",
    "quality assurance": "qa",
    "customer support": "customer service",
    "customers support": "customer service",
    "js": "javascript",
}

ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")                      # QA, API, ETL, SQL, HTTP, REST, CRM...
CODEY_PATTERN   = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)+")  # AZ-104, DHIS2-like, etc.

def _normalize_phrase(words):
    # words is a list of spaCy tokens; turn into a normalized phrase of lemmas
    lemmas = []
    for t in words:
        if t.is_space or t.is_punct:
            continue
        # keep acronyms & short tokens (QA, API, ETL) – do not filter by length
        lemma = t.lemma_.lower().strip()
        if lemma and lemma not in stop_words:
            lemmas.append(lemma)
    if not lemmas:
        return ""
    phrase = " ".join(lemmas)
    phrase = SYNONYMS.get(phrase, phrase)
    phrase = re.sub(r"\s{2,}", " ", phrase)  # collapse repeated spaces
    return phrase

def extract_terms(text: str) -> set:
    doc = nlp(text)
    terms = set()

    # 1) Single tokens
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        if tok.ent_type_ in {"DATE", "TIME", "MONEY", "ORDINAL", "CARDINAL"}:
            continue

        # acronyms (QA, API, ETL, SQL, HTTP, REST...)
        if ACRONYM_PATTERN.fullmatch(tok.text):
            terms.add(tok.text.lower())
            continue

        # hyphenated/code-like tokens (AZ-104, DHIS2-like)
        if CODEY_PATTERN.fullmatch(tok.text):
            terms.add(tok.text.lower())
            continue

        # nouns/proper nouns (skills/tools)
        if tok.pos_ in {"NOUN", "PROPN"}:
            lemma = tok.lemma_.lower().strip()
            if lemma and lemma not in stop_words:
                terms.add(SYNONYMS.get(lemma, lemma))
            continue

        # selected verbs (QA/support concepts)
        if tok.pos_ == "VERB":
            lemma = tok.lemma_.lower().strip()
            if lemma in ALLOWED_VERBS:
                terms.add(lemma)
                # also add -ing nouny form
                if lemma.endswith("e"):
                    terms.add(lemma[:-1] + "ing")
                else:
                    terms.add(lemma + "ing")

    # 2) Noun chunks (short phrases)
    for chunk in doc.noun_chunks:
        words = [t for t in chunk if not t.is_space and not t.is_punct]
        # strip leading/trailing stopwords
        while words and (words[0].is_stop or words[0].is_space or words[0].is_punct):
            words = words[1:]
        while words and (words[-1].is_stop or words[-1].is_space or words[-1].is_punct):
            words = words[:-1]
        if not words:
            continue
        # keep short phrases to avoid long tails
        if 1 <= len(words) <= 4:
            phrase = _normalize_phrase(words)
            if phrase:
                terms.add(phrase)

    # 3) Post-normalization folding
    normalized = set()
    for t in terms:
        t2 = SYNONYMS.get(t, t)
        t2 = re.sub(r"\s{2,}", " ", t2.strip())
        if t2:
            normalized.add(t2)

    return normalized

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

    # --- Dynamic JD-driven Keyword/Phrase Matching ---
    jd_terms = extract_terms(jd_text)
    resume_terms = extract_terms(resume_text)

    matched_keywords = sorted(jd_terms.intersection(resume_terms))
    missing_keywords = sorted(jd_terms.difference(resume_terms))

    match_score = round((len(matched_keywords) / len(jd_terms) * 100), 2) if jd_terms else 0.0

    # Results
    st.subheader("🔍 Resume vs JD Keyword/Phrase Match")
    st.markdown(f"**Match Score:** {match_score}%")
    st.markdown(f"**Matched ({len(matched_keywords)}):** `{', '.join(matched_keywords)}`")
    st.markdown(f"**Missing ({len(missing_keywords)}):** `{', '.join(missing_keywords)}`")

    # Suggestions
    st.subheader("💡 Copilot-style Suggestions")
    if missing_keywords:
        st.markdown("• Consider adding the following keywords/phrases to better align with the JD:")
        st.markdown(f"`{', '.join(missing_keywords)}`")
    elif jd_terms:
        st.markdown("• ✅ Your resume already covers all the key terms in the job description!")
    else:
        st.markdown("• (No clear keywords detected in the JD text. Try pasting the full JD.)")

    if matched_keywords:
        st.markdown("• Your resume already includes:")
        st.markdown(f"`{', '.join(matched_keywords)}`")

    st.markdown("• Tip: Reflect these in your **Experience** bullets and **Skills** section for stronger ATS visibility.")

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