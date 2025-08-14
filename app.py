import streamlit as st
import fitz  # PyMuPDF
import io
import re
import csv
import nltk
import spacy
import os, json, yaml, time
from typing import Tuple, Dict, Any
from nltk.corpus import stopwords
from docx import Document
from striprtf.striprtf import rtf_to_text
from rapidfuzz import fuzz

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="AI Resume & Job Analyzer",
    page_icon="📄",
    layout="centered",
)

# ---------------- Sidebar ----------------
st.sidebar.header("📌 How to Use")
st.sidebar.markdown("""
1. **Upload your Resume** (PDF, DOCX, RTF, or TXT).
2. **Paste the Job Description (JD)** into the text box.
3. Click **Analyze** to see:
   - ✅ Simple match score  
   - 📈 Weighted (ATS-style) score  
   - 🗂 Matched / ❌ Missing keywords & phrases  
4. Expand details to view **per-term breakdown** or **download CSVs**.
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<a href="https://github.com/Griffindeetox/AI-ATS-resume-job-analyzer" target="_blank">💻 <b>View Source Code on GitHub</b></a>',
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 *Made by Adeyemi O*")

# ---------------- Title & tagline ----------------
st.title("🚀 AI Resume & Job Match Analyzer (Beta)")
st.markdown(
    "<p style='color: gray; font-size: 16px;'>Upload your resume & paste the job description to get instant simple and ATS-style weighted scores, with matched/missing terms.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------- Load NLP tools ----------------
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# =========================================================
# ============ Auto-updating Configs (NEW) ================
# =========================================================
CFG_DIR = "config"
DATA_DIR = "data"

WEIGHTS_PATH  = os.path.join(CFG_DIR, "weights.yaml")
PROFILES_PATH = os.path.join(CFG_DIR, "profiles.json")
SYN_BASE_PATH = os.path.join(DATA_DIR, "synonyms.json")          # your base seed
SYN_USER_PATH = os.path.join(DATA_DIR, "synonyms.user.json")     # learned via UI
STOP_PATH     = os.path.join(DATA_DIR, "stop_phrases.json")
TELEMETRY_PATH= os.path.join(DATA_DIR, "telemetry.jsonl")

# Built-in minimum fallbacks so the app never crashes if files are missing
BUILTIN_WEIGHTS = {
    "weights": {"critical": 3.0, "important": 2.0, "nice": 1.0},
    "thresholds": {"critical": 88, "important": 82, "nice": 75},
    "guardrails": {"min_critical_pct": 55.0, "max_gap": 35.0},
    "profiles": {
        "rf": {"weights": {"critical": 3.5}, "thresholds": {"critical": 90, "important": 84, "nice": 75}},
        "qa": {"weights": {"critical": 3.2}, "thresholds": {"critical": 88, "important": 83, "nice": 75}},
    },
}
BUILTIN_PROFILES = {
    "profiles": {
        "rf": {
            "markers": ["rf","radio","digital radio","microwave","antenna","vswr","v-swr","telephony","radio interference","rf optimization","drive test"],
            "critical_add": ["rf","digital radio","microwave","antenna","vswr"]
        },
        "qa": {
            "markers": ["qa","test case","manual testing","automation testing","selenium","cypress","postman","testrail","test plan","bug report"],
            "critical_add": ["qa","manual testing","test case"]
        }
    }
}
BUILTIN_SYNONYMS = {
    "quality assurance":"qa","qa testing":"qa","manual software qa testing":"qa",
    "application programming interface":"api","web service":"api","soap api":"api","rest apis":"rest api",
    "db":"database","data store":"database","sql database":"database",
    "postgres":"postgresql","postgre sql":"postgresql",
    "js":"javascript",
    "customer support":"customer service","customers support":"customer service",
    "end user":"customer","client":"customer","stakeholder":"customer",
    "docs":"documentation","manual":"documentation","guides":"documentation","technical writing":"documentation",
    "connect":"integration","api link":"integration","system integration":"integration",
    "data transformation":"etl","data pipeline":"etl",
    "radio frequency":"rf","radio-frequency":"rf","v-swr":"vswr","microwave link":"microwave backhaul"
}
BUILTIN_STOP = [
    "3 5 year","3-5 years","3 to 5 years","bonus","audience","new","lot","high level",
    "fully computer literate","driving license","availability to travel","security clearance","standby rotation",
    "job description","responsibilities include","preferred qualifications","nice to have","about the company",
    "our mission","perks and benefits","how to apply","contact us","location","requirements","benefits"
]

def _safe_read_json(path: str, default: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _safe_read_yaml(path: str, default: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    except Exception:
        pass
    return default

def _ensure_files():
    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):   open(WEIGHTS_PATH, "w").write(yaml.safe_dump(BUILTIN_WEIGHTS, sort_keys=False))
    if not os.path.exists(PROFILES_PATH):  open(PROFILES_PATH, "w").write(json.dumps(BUILTIN_PROFILES, indent=2))
    if not os.path.exists(SYN_BASE_PATH):  open(SYN_BASE_PATH, "w").write(json.dumps(BUILTIN_SYNONYMS, indent=2))
    if not os.path.exists(SYN_USER_PATH):  open(SYN_USER_PATH, "w").write(json.dumps({}, indent=2))
    if not os.path.exists(STOP_PATH):      open(STOP_PATH, "w").write(json.dumps(BUILTIN_STOP, indent=2))
    if not os.path.exists(TELEMETRY_PATH): open(TELEMETRY_PATH, "a").close()

def _load_configs():
    weights = _safe_read_yaml(WEIGHTS_PATH, BUILTIN_WEIGHTS)
    profiles = _safe_read_json(PROFILES_PATH, BUILTIN_PROFILES)
    syn_base = _safe_read_json(SYN_BASE_PATH, BUILTIN_SYNONYMS)
    syn_user = _safe_read_json(SYN_USER_PATH, {})
    synonyms = {**syn_base, **syn_user}
    stops = _safe_read_json(STOP_PATH, BUILTIN_STOP)
    mtimes = {
        "weights": os.path.getmtime(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else 0,
        "profiles":os.path.getmtime(PROFILES_PATH) if os.path.exists(PROFILES_PATH) else 0,
        "syn_base":os.path.getmtime(SYN_BASE_PATH) if os.path.exists(SYN_BASE_PATH) else 0,
        "syn_user":os.path.getmtime(SYN_USER_PATH) if os.path.exists(SYN_USER_PATH) else 0,
        "stops":  os.path.getmtime(STOP_PATH) if os.path.exists(STOP_PATH) else 0,
    }
    return weights, profiles, synonyms, stops, mtimes

def _hot_reload_if_changed(state_key="cfg_mtimes"):
    mtimes_now = {
        "weights": os.path.getmtime(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else 0,
        "profiles":os.path.getmtime(PROFILES_PATH) if os.path.exists(PROFILES_PATH) else 0,
        "syn_base":os.path.getmtime(SYN_BASE_PATH) if os.path.exists(SYN_BASE_PATH) else 0,
        "syn_user":os.path.getmtime(SYN_USER_PATH) if os.path.exists(SYN_USER_PATH) else 0,
        "stops":  os.path.getmtime(STOP_PATH) if os.path.exists(STOP_PATH) else 0,
    }
    old = st.session_state.get(state_key)
    if old is None or any(mtimes_now[k] != old.get(k, -1) for k in mtimes_now):
        (st.session_state["weights_cfg"],
         st.session_state["profiles_cfg"],
         st.session_state["synonyms"],
         st.session_state["stop_phrases"],
         st.session_state[state_key]) = _load_configs()
        st.toast("Configs reloaded ✅", icon="🔄")

# Ensure config files, load once, and hot-reload on changes
_ensure_files()
if "weights_cfg" not in st.session_state:
    (st.session_state["weights_cfg"],
     st.session_state["profiles_cfg"],
     st.session_state["synonyms"],
     st.session_state["stop_phrases"],
     st.session_state["cfg_mtimes"]) = _load_configs()
_hot_reload_if_changed()

# Sidebar admin quick action
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reload configs"):
    (st.session_state["weights_cfg"],
     st.session_state["profiles_cfg"],
     st.session_state["synonyms"],
     st.session_state["stop_phrases"],
     st.session_state["cfg_mtimes"]) = _load_configs()
    st.success("Configs reloaded from disk.")

# Handy local variables
WEIGHTS_CFG  = st.session_state["weights_cfg"]
PROFILES_CFG = st.session_state["profiles_cfg"]
SYNONYMS_MAP = st.session_state["synonyms"]
STOP_PHRASES = set(map(str.lower, st.session_state["stop_phrases"]))

# ---------------- File text extractors ----------------
def extract_text_from_pdf(file_obj) -> str:
    with fitz.open(stream=file_obj.read(), filetype="pdf") as doc:
        text = []
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def extract_text_from_docx(file_obj) -> str:
    data = file_obj.read()
    bio = io.BytesIO(data)
    doc = Document(bio)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_rtf(file_obj) -> str:
    data = file_obj.read()
    try:
        return rtf_to_text(data.decode("utf-8", errors="ignore"))
    except Exception:
        return rtf_to_text(data.decode("latin-1", errors="ignore"))

def extract_text_from_txt(file_obj) -> str:
    data = file_obj.read()
    try:
        text = data.decode("utf-8")
    except Exception:
        text = data.decode("latin-1", errors="ignore")
    text = re.sub(r"[#*_>`~\-]{1,}", " ", text)  # light formatting cleanup
    return text

def extract_text_from_any(file_obj, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_obj)
    if name.endswith(".docx"):
        return extract_text_from_docx(file_obj)
    if name.endswith(".rtf"):
        return extract_text_from_rtf(file_obj)
    return extract_text_from_txt(file_obj)

# ---------------- Dynamic term extraction (tokens + short phrases + acronyms) ----------------
ALLOWED_VERBS = {
    "test", "testing", "troubleshoot", "troubleshooting", "debug", "debugging",
    "integrate", "integration", "document", "documentation", "support",
    "analyze", "analysis"
}

ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")                      # QA, API, ETL, SQL, HTTP, REST, CRM...
CODEY_PATTERN   = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)+")

def _normalize_phrase(words):
    lemmas = []
    for t in words:
        if t.is_space or t.is_punct:
            continue
        lemma = t.lemma_.lower().strip()
        if lemma and lemma not in stop_words and lemma not in STOP_PHRASES:
            lemmas.append(lemma)
    if not lemmas:
        return ""
    phrase = " ".join(lemmas)
    phrase = SYNONYMS_MAP.get(phrase, phrase)
    phrase = re.sub(r"\s{2,}", " ", phrase)
    return phrase

def normalize_term(t: str) -> str:
    t = t.lower().strip()
    return SYNONYMS_MAP.get(t, t)

def extract_terms(text: str) -> set:
    # remove global stop phrases up-front
    low = text.lower()
    for p in STOP_PHRASES:
        low = low.replace(p, " ")
    doc = nlp(low)
    terms = set()

    # 1) Single tokens
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        if tok.ent_type_ in {"DATE", "TIME", "MONEY", "ORDINAL", "CARDINAL"}:
            continue

        if ACRONYM_PATTERN.fullmatch(tok.text):           # QA, API, ETL, SQL, HTTP, REST...
            terms.add(tok.text.lower())
            continue

        if CODEY_PATTERN.fullmatch(tok.text):             # AZ-104, DHIS2-like
            terms.add(tok.text.lower())
            continue

        if tok.pos_ in {"NOUN", "PROPN"}:
            lemma = tok.lemma_.lower().strip()
            if lemma and lemma not in stop_words and lemma not in STOP_PHRASES:
                terms.add(SYNONYMS_MAP.get(lemma, lemma))
            continue

        if tok.pos_ == "VERB":
            lemma = tok.lemma_.lower().strip()
            if lemma in ALLOWED_VERBS and lemma not in STOP_PHRASES:
                terms.add(lemma)
                if lemma.endswith("e"):
                    terms.add(lemma[:-1] + "ing")
                else:
                    terms.add(lemma + "ing")

    # 2) Noun chunks (short phrases)
    for chunk in doc.noun_chunks:
        words = [t for t in chunk if not t.is_space and not t.is_punct]
        while words and (words[0].is_stop or words[0].is_space or words[0].is_punct):
            words = words[1:]
        while words and (words[-1].is_stop or words[-1].is_space or words[-1].is_punct):
            words = words[:-1]
        if not words:
            continue
        if 1 <= len(words) <= 4:
            phrase = _normalize_phrase(words)
            if phrase:
                terms.add(phrase)

    # 3) Post-normalization folding
    normalized = set()
    for t in terms:
        t2 = SYNONYMS_MAP.get(t, t)
        t2 = re.sub(r"\s{2,}", " ", t2.strip())
        if t2 and t2 not in STOP_PHRASES:
            normalized.add(t2)

    return normalized

# ---------------- Matching helpers (exact, synonym, fuzzy) ----------------
CATEGORY_HINTS = {
    # Critical JD concepts for QA/API roles (extendable via profiles later)
    "qa": "critical",
    "manual testing": "critical",
    "test case": "critical",
    "rest api": "critical",
    "api": "critical",
    "http status": "critical",
    "javascript": "critical",
    "integration": "critical",
    "data integration": "critical",
    "etl": "critical",
    "sql": "critical",
    "postgresql": "critical",

    # Important but not always must-have
    "documentation": "important",
    "customer service": "important",
    "troubleshooting": "important",
    "debugging": "important",
    "salesforce": "important",
    "dhis2": "important",
    "commcare": "important",
    "kobo toolbox": "important",
    "openmrs": "important",
    "remote": "important",
    "remote-first": "important",
}

def expand_synonyms(term: str) -> set:
    base = normalize_term(term)
    variants = {base}
    # reverse lookup
    for k, v in SYNONYMS_MAP.items():
        if v == base:
            variants.add(k)
    if not base.endswith("s"):
        variants.add(base + "s")
    else:
        variants.add(base.rstrip("s"))
    return {normalize_term(v) for v in variants}

# Build scoring config from weights.yaml (instead of hard-coded)
def get_scoring_config():
    w = WEIGHTS_CFG.get("weights", {})
    t = WEIGHTS_CFG.get("thresholds", {})
    return {
        "weights": {
            "critical": float(w.get("critical", 3.0)),
            "important": float(w.get("important", 2.0)),
            "nice": float(w.get("nice", 1.0)),
        },
        "thresholds": {
            "critical": int(t.get("critical", 88)),
            "important": int(t.get("important", 82)),
            "nice": int(t.get("nice", 75)),
        }
    }

# Precompiled token regex (FIXED: safe dash placement/escaping)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\/ ]{1,40}")

def any_exact_or_fuzzy_match(term: str, resume_terms: set, resume_text: str, threshold: int) -> tuple[bool, str]:
    """
    Tries exact match, synonym exact, then fuzzy against resume_terms and resume_text.
    Returns (matched, method) where method in {"exact","synonym","fuzzy-terms","fuzzy-text","no"}.
    """
    base = normalize_term(term)
    syns = expand_synonyms(base)

    # 1) exact in extracted terms
    if base in resume_terms:
        return True, "exact"
    if syns.intersection(resume_terms):
        return True, "synonym"

    # 2) fuzzy vs extracted terms
    for rt in resume_terms:
        if fuzz.partial_ratio(base, rt) >= threshold:
            return True, "fuzzy-terms"
        for s in syns:
            if fuzz.partial_ratio(s, rt) >= threshold:
                return True, "fuzzy-terms"

    # 3) fuzzy vs raw text (backup)
    tokens = set(TOKEN_RE.findall(resume_text.lower()))
    for tok in tokens:
        if fuzz.partial_ratio(base, tok) >= threshold:
            return True, "fuzzy-text"
        for s in syns:
            if fuzz.partial_ratio(s, tok) >= threshold:
                return True, "fuzzy-text"

    return False, "no"

def categorize_term(term: str) -> str:
    t = normalize_term(term)
    if t in CATEGORY_HINTS:
        return CATEGORY_HINTS[t]
    if any(k in t for k in ["qa","rest api","api","http status","javascript","etl","integration","sql","postgresql","test case"]):
        return "critical"
    if any(k in t for k in ["troubleshoot","debug","documentation","customer service","salesforce","dhis2","commcare","kobo toolbox","openmrs","remote"]):
        return "important"
    return "nice"

def score_weighted(jd_terms: set, resume_terms: set, resume_text: str):
    SCORING_CONFIG = get_scoring_config()
    items = []
    total_possible = 0.0
    total_earned = 0.0

    for term in sorted(jd_terms):
        category = categorize_term(term)
        weight = SCORING_CONFIG["weights"][category]
        threshold = SCORING_CONFIG["thresholds"][category]

        matched, method = any_exact_or_fuzzy_match(term, resume_terms, resume_text, threshold)

        earned = weight if matched else 0.0
        total_possible += weight
        total_earned += earned

        items.append({
            "Term": term,
            "Category": category,
            "Matched": "✅" if matched else "❌",
            "Method": method if matched != "no" else "-",
            "Weight": round(weight, 2),
            "Earned": round(earned, 2),
        })

    score = round((total_earned / total_possible) * 100, 2) if total_possible > 0 else 0.0
    return score, items, total_possible, total_earned

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
    resume_text = extract_text_from_any(resume_file, resume_file.name)

    # Previews
    st.subheader("📄 Resume Preview")
    st.text_area("Your Resume Content", resume_text, height=160)

    st.subheader("🧾 Job Description Preview")
    st.text_area("Job Description Content", jd_text, height=160)

    # --- Extract terms
    jd_terms = extract_terms(jd_text)
    resume_terms = extract_terms(resume_text)

    # --- Simple (unweighted) match for reference
    simple_matched = sorted(jd_terms.intersection(resume_terms))
    simple_missing = sorted(jd_terms.difference(resume_terms))
    simple_score = round((len(simple_matched) / len(jd_terms) * 100), 2) if jd_terms else 0.0

    # --- Weighted score
    weighted_score, items, total_possible, total_earned = score_weighted(jd_terms, resume_terms, resume_text)

    # Top summary (two scores side-by-side)
    st.subheader("📊 Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Simple Match", f"{simple_score}%")
    with col2:
        st.metric("Weighted Match (ATS-style)", f"{weighted_score}%")

    # Simple details (toggle + CSV)
    st.subheader("🔍 Simple Match (for reference)")
    show_simple = st.toggle("Show simple match details", value=False, key="simple_details_toggle")
    if show_simple:
        st.markdown(f"**Matched ({len(simple_matched)}):** `{', '.join(simple_matched)}`")
        st.markdown(f"**Missing ({len(simple_missing)}):** `{', '.join(simple_missing)}`")

        # CSV download for simple match lists
        if simple_matched or simple_missing:
            import io as _io
            buf = _io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["Term", "Status"])
            for t in simple_matched:
                writer.writerow([t, "Matched"])
            for t in simple_missing:
                writer.writerow([t, "Missing"])
            st.download_button(
                "Download simple match as CSV",
                buf.getvalue(),
                file_name="simple_match.csv",
                mime="text/csv"
            )

    # Weighted breakdown + grouping
    st.subheader("📈 Weighted Match (ATS-style)")
    st.caption("Critical terms count more than nice-to-have ones. Synonyms and fuzzy matches are accepted.")

    # Sort by importance (weight desc) then matched
    items_sorted = sorted(items, key=lambda d: (d["Weight"], d["Matched"]), reverse=True)

    # Group into Critical / Important / Nice (matched/missing)
    groups = {"critical": {"matched": [], "missing": []},
              "important": {"matched": [], "missing": []},
              "nice": {"matched": [], "missing": []}}

    for row in items_sorted:
        bucket = row["Category"]
        (groups[bucket]["matched"] if row["Matched"] == "✅" else groups[bucket]["missing"]).append(row["Term"])

    # Expanders per group
    for bucket, label in [("critical", "Critical"), ("important", "Important"), ("nice", "Nice to Have")]:
        with st.expander(f"{label} — Matched ({len(groups[bucket]['matched'])}) / Missing ({len(groups[bucket]['missing'])})", expanded=False):
            if groups[bucket]["matched"]:
                st.markdown(f"**Matched:** `{', '.join(sorted(groups[bucket]['matched']))}`")
            if groups[bucket]["missing"]:
                st.markdown(f"**Missing:** `{', '.join(sorted(groups[bucket]['missing']))}`")

    # Details toggle + full breakdown CSV
    show_details = st.toggle("Show detailed breakdown", value=False)
    if show_details:
        st.markdown("**Term Impact Breakdown**")
        st.table(items_sorted)

        if items_sorted:
            import io as _io2
            buf2 = _io2.StringIO()
            writer2 = csv.DictWriter(buf2, fieldnames=list(items_sorted[0].keys()))
            writer2.writeheader()
            writer2.writerows(items_sorted)
            st.download_button(
                "Download breakdown as CSV",
                buf2.getvalue(),
                file_name="ats_breakdown.csv",
                mime="text/csv"
            )

    # Suggestions (weighted perspective)
    missing_weighted = [d["Term"] for d in items if d["Matched"] == "❌"]
    st.subheader("💡 Copilot-style Suggestions")
    if missing_weighted:
        # Prioritize by category
        critical_missing = [t for t in missing_weighted if categorize_term(t) == "critical"]
        important_missing = [t for t in missing_weighted if categorize_term(t) == "important"]

        if critical_missing:
            st.markdown("• **Top priority (Critical):**")
            st.markdown(f"`{', '.join(sorted(set(critical_missing)))}`")
        if important_missing:
            st.markdown("• **Next (Important):**")
            st.markdown(f"`{', '.join(sorted(set(important_missing)))}`")

        # Everything else
        other_missing = sorted(set(missing_weighted) - set(critical_missing) - set(important_missing))
        if other_missing:
            st.markdown("• **Nice to have:**")
            st.markdown(f"`{', '.join(other_missing)}`")
    elif jd_terms:
        st.markdown("• ✅ Your resume already covers the key terms in this JD (consider adding measurable results/examples).")
    else:
        st.markdown("• (No clear keywords detected in the JD text. Try pasting the full JD.)")

    # Learn synonyms (saved into data/synonyms.user.json)
    with st.expander("Mark a missing JD term as equivalent to a resume term (learn a synonym)"):
        miss = st.text_input("Missing term (from JD):", key="miss_term")
        present = st.text_input("Equivalent term in your resume:", key="present_term")
        if st.button("Save synonym"):
            if miss and present:
                try:
                    user_syn = _safe_read_json(SYN_USER_PATH, {})
                    user_syn[miss.lower()] = present.lower()
                    with open(SYN_USER_PATH, "w", encoding="utf-8") as f:
                        json.dump(user_syn, f, indent=2)
                    st.success(f"Saved: '{miss}' ≈ '{present}'.")
                    # Refresh merged synonyms in-session
                    base_syn = _safe_read_json(SYN_BASE_PATH, BUILTIN_SYNONYMS)
                    st.session_state["synonyms"] = {**base_syn, **user_syn}
                    st.session_state["cfg_mtimes"]["syn_user"] = os.path.getmtime(SYN_USER_PATH)
                except Exception as e:
                    st.error(f"Could not save synonym: {e}")

    # Final tip
    st.markdown("• Tip: Put **critical terms** into **recent Experience bullets** and **Skills** for better ATS visibility.")

else:
    st.info("Please upload a resume and paste the job description to start the analysis.")

# ---------------- Footer ----------------
st.markdown(
    """
    <style>
      .app-footer { text-align:center; opacity:0.7; margin-top:1.0rem; }
    </style>
    <div class="app-footer">👨‍💻 Made by <b>Adeyemi O</b></div>
    """,
    unsafe_allow_html=True
)