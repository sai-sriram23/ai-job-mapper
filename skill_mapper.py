import json
import os
from groq import Groq

# Load environment variables from .env if present
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"\'')

# ----------------------------
# Configure Groq
# ----------------------------
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


JSON_FILE = "role_skills.json"

# ----------------------------
# Create JSON
# ----------------------------
def initialize_json():
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, "w") as f:
            json.dump({}, f, indent=4)

# ----------------------------
# Load JSON
# ----------------------------
def load_database():
    initialize_json()

    with open(JSON_FILE, "r") as f:
        return json.load(f)

# ----------------------------
# Save JSON
# ----------------------------
def save_database(data):
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ----------------------------
# Call Groq AI
# ----------------------------
def fetch_skills_from_ai(job_role):

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

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# ----------------------------
# Get Skills
# ----------------------------
def get_role_skills(job_role):

    db = load_database()

    # Case-insensitive match check
    for key in db.keys():
        if key.lower() == job_role.lower():
            print("Loaded from JSON (case-insensitive)")
            return db[key]["required_skills"]

    # Normalize new role to title case
    normalized_role = job_role.title()
    if normalized_role in db:
        print("Loaded from JSON")
        return db[normalized_role]["required_skills"]

    print("Fetching from Groq...")

    ai_result = fetch_skills_from_ai(normalized_role)

    db[normalized_role] = ai_result

    save_database(db)

    return ai_result["required_skills"]

# ----------------------------
# Compare Skills
# ----------------------------
def compare_skills(user_skills, required_skills):

    user = {skill.lower() for skill in user_skills}
    required = {skill.lower() for skill in required_skills}

    matched = sorted(user & required)
    missing = sorted(required - user)

    return matched, missing


# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":

    role = "Machine Learning Engineer"

    user_skills = [
        "Python",
        "SQL",
        "Git"
    ]

    required = get_role_skills(role)

    matched, missing = compare_skills(
        user_skills,
        required
    )

    print("\nRequired Skills")
    print(required)

    print("\nMatched Skills")
    print(matched)

    print("\nMissing Skills")
    print(missing)