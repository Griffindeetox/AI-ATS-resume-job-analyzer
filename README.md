# 🚀 AI Resume & Job Match Analyzer

An open-source **AI-powered ATS resume checker** that analyzes your resume against a job description to give you:
- ✅ Match Score  
- 🗂 Matched Keywords  
- ❌ Missing Keywords  
- 💡 Smart Suggestions

This project is designed to **mimic modern ATS (Applicant Tracking Systems)** so you can optimize your resume for better job application results.

---

## 🎥 Demo

![App Demo](https://github.com/user-attachments/assets/bffc9900-e909-424f-9f1b-3230bdbce286)

---

## ✨ Features

- **Multi-format Resume Upload** — Supports `.pdf`, `.docx`, `.txt`
- **JD Paste Box** — No need to upload the job description; just paste it in
- **Keyword Extraction & Matching** — Finds both matched and missing skills
- **Match Score** — Quickly see how well your resume fits the JD
- **Smart Suggestions** — Copilot-style tips for improving your resume

---

## 📌 How to Use

1. **Upload your Resume** (`.pdf`, `.docx`, or `.txt`)
2. **Paste the Job Description** into the provided box
3. Click **Analyze**
4. View:
   - Match Score  
   - Matched Skills  
   - Missing Skills  
5. Improve your resume based on the suggestions

---

## 🛠 Installation

```bash
# Clone the repo
git clone https://github.com/Griffindeetox/AI-ATS-resume-job-analyzer.git
cd AI-ATS-resume-job-analyzer

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
