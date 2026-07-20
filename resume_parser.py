import pdfplumber
import docx
import json
import os
from groq import Groq

# Ensure environment variables are loaded
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"\'')

# -----------------------------
# Configure Groq client
# -----------------------------
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

JSON_FILE = "role_skills.json"

# -----------------------------
# Database helpers
# -----------------------------
def load_skills_db():
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, "w") as f:
            json.dump({}, f, indent=4)
            
    with open(JSON_FILE, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def save_skills_db(data):
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)

# -----------------------------
# Fetch standard technical skills for a role via AI
# -----------------------------
def fetch_skills_for_role_ai(job_role):
    prompt = f"""
You are an expert career advisor.

For the job role:
{job_role}

Return ONLY valid JSON.

Format:

{{
    "required_skills":[
        "Skill1",
        "Skill2"
    ]
}}

Rules:
- Return exactly 10 technical skills.
- No explanation.
- No markdown.
- No extra text.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"required_skills": []}

# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_pdf_text(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# -----------------------------
# Extract text from DOCX
# -----------------------------
def extract_docx_text(uploaded_file):
    document = docx.Document(uploaded_file)
    text = ""
    for para in document.paragraphs:
        text += para.text + "\n"
    return text

# -----------------------------
# Main Text Extractor
# -----------------------------
def extract_resume_text(uploaded_file):
    if isinstance(uploaded_file, str):
        filename = uploaded_file
    elif hasattr(uploaded_file, "name"):
        filename = uploaded_file.name
    else:
        raise ValueError("Invalid file input type. Must be a path string or file-like object.")

    if filename.lower().endswith(".pdf"):
        return extract_pdf_text(uploaded_file)
    elif filename.lower().endswith(".docx"):
        return extract_docx_text(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Upload PDF or DOCX.")

# -----------------------------
# Analyze Resume & Self-Update database if role is new
# -----------------------------
def analyze_resume_profile(uploaded_file):
    # 1. Parse text from the file
    resume_text = extract_resume_text(uploaded_file)
    
    # 2. Load all existing technical skills from database as standard references
    db = load_skills_db()
    all_ref_skills = set()
    for role, data in db.items():
        for skill in data.get("required_skills", []):
            all_ref_skills.add(skill)
    ref_skills_str = ", ".join(sorted(list(all_ref_skills))) if all_ref_skills else "Python, SQL, Java, React, HTML, CSS"

    # 3. Call AI to extract technical skills from resume text
    prompt = f"""
You are an expert ATS Resume Analyzer.

Extract ONLY technical skills from the resume.

Return ONLY JSON.

Format:

{{
    "skills": [
        "Python",
        "SQL",
        "Machine Learning"
    ]
}}

Reference Skills (Use this list to guide your technical skill extraction):
{ref_skills_str}

Rules:
- Extract only technical skills.
- Remove duplicate skills.
- Ignore soft skills, education, projects, certifications.
- No explanation.

Resume:
{resume_text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        skills = result.get("skills", [])
    except Exception:
        skills = []

    return {
        "skills": skills,
        "resume_text": resume_text
    }


def suggest_new_role_ai(resume_text, resume_skills):
    prompt = f"""
You are an expert AI Career Advisor.
The candidate's profile and resume do not match any of our standard career categories (Analyst, Data Scientist, Software Engineer, Web Developer, DevOps, etc.).

Analyze the candidate's resume content and technical skills to suggest a single, most suitable specialized job role. Make sure the suggested role explicitly states the broader parent category it belongs to if it is a sub-role (e.g. "Accessibility Specialist (under Web Developer)" or "Frameworks Specialist (under Software Engineer)").

Also, provide a list of exactly 10 standard technical skills required for this recommended job role.

Return ONLY JSON.

Format:
{{
    "suggested_role": "UI/UX Designer",
    "required_skills": [
        "User Research",
        "Wireframing",
        "Prototyping",
        "Visual Design",
        "Interaction Design",
        "User Experience",
        "Human-Computer Interaction",
        "Design Systems",
        "Front-end Development",
        "Adobe Creative Suite"
    ]
}}

Candidate's Extracted Skills:
{", ".join(resume_skills)}

Resume Content:
{resume_text}
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {
            "suggested_role": "Software Engineer",
            "required_skills": ["Python", "Algorithms", "Data Structures", "System Design", "SQL", "Git"]
        }