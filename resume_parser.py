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

import re

# -----------------------------
# Common Technical Skills Dictionary for Fallback Extraction
# -----------------------------
COMMON_TECH_SKILLS = [
    "Python", "Java", "C++", "C#", "C", "R", "Go", "Golang", "Rust", "Swift", "Kotlin", "PHP", "Ruby", "Scala",
    "HTML", "HTML5", "CSS", "CSS3", "JavaScript", "TypeScript", "React", "React.js", "Angular", "Vue", "Vue.js", "Next.js", "Node.js", "Express", "Bootstrap", "Tailwind", "Sass", "jQuery",
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "Redis", "Cassandra", "DynamoDB", "Firebase",
    "Django", "Flask", "FastAPI", "Spring Boot", ".NET", "Laravel",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "DevOps", "Terraform", "Ansible", "CI/CD", "Jenkins", "Git", "GitHub", "GitLab", "Linux", "Bash", "Shell",
    "Machine Learning", "Deep Learning", "Artificial Intelligence", "NLP", "Natural Language Processing", "Computer Vision", "TensorFlow", "PyTorch", "Keras", "Scikit-Learn", "Pandas", "NumPy", "Matplotlib", "Seaborn", "OpenCV", "Power BI", "Tableau", "Excel",
    "Data Structures", "Algorithms", "DSA", "System Design", "OOP", "REST API", "GraphQL", "Microservices",
    "Figma", "UI/UX", "Wireframing", "Prototyping", "Agile", "Scrum", "Jira", "PyTest", "Selenium"
]

def extract_skills_fallback(resume_text, ref_skills_list=None):
    """Fallback skill extractor using regex keyword matching if LLM API is unavailable."""
    if not resume_text:
        return []
    skills_found = set()
    all_target_skills = list(set(COMMON_TECH_SKILLS + (ref_skills_list or [])))
    
    for skill in all_target_skills:
        # Match skill using regex boundaries to prevent partial word mismatches
        escaped_skill = re.escape(skill)
        pattern = r'(?<![A-Za-z0-9])' + escaped_skill + r'(?![A-Za-z0-9])'
        if re.search(pattern, resume_text, re.IGNORECASE):
            skills_found.add(skill)
            
    return sorted(list(skills_found))

# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_pdf_text(uploaded_file):
    text = ""
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"pdfplumber extraction error: {e}")
        
    # If pdfplumber returned empty text, try pypdf as fallback
    if not text.strip():
        try:
            import pypdf
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            reader = pypdf.PdfReader(uploaded_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"pypdf extraction error: {e}")

    return text

# -----------------------------
# Extract text from DOCX
# -----------------------------
def extract_docx_text(uploaded_file):
    text = ""
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        document = docx.Document(uploaded_file)
        for para in document.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"docx extraction error: {e}")
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
    ref_skills_list = list(all_ref_skills)
    ref_skills_str = ", ".join(sorted(ref_skills_list)) if ref_skills_list else "Python, SQL, Java, React, HTML, CSS"

    skills = []

    # 3. Call Groq AI to extract technical skills from resume text
    if resume_text.strip():
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
        except Exception as e:
            print(f"Groq API skill extraction error: {e}")
            skills = []

    # 4. Fallback to Regex keyword extraction if Groq API produced no skills or failed
    if not skills and resume_text.strip():
        skills = extract_skills_fallback(resume_text, ref_skills_list)

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