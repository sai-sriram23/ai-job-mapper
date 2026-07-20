import os
import json
from tavily import TavilyClient
from groq import Groq

# Ensure environment variables are loaded
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"\'')

# Configure clients
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def _sanitize_text(text):
    """Remove non-ASCII characters that crash Windows charmap codec."""
    if not text:
        return ""
    return text.encode("ascii", errors="ignore").decode("ascii")

def get_job_market_insights(job_role):
    # 1. Query Tavily for live facts
    query = f"current job market trends, average salary range, top companies hiring, required certifications, project ideas, and interview questions for: {job_role}"
    try:
        search_results = tavily_client.search(query=query, search_depth="basic")
        results_text = "\n".join([_sanitize_text(r.get("content", "")) for r in search_results.get("results", [])])
    except Exception as e:
        results_text = f"Failed to fetch live search results: {str(e)}"
        
    # 2. Use Groq to structure the search results into a clean JSON format
    prompt = f"""
You are an expert career advisor and research analyst.
Below are some live search results about the job market for the role of '{job_role}'.

Summarize and structure this information into a valid JSON object.

Format:
{{
    "average_salary": "$90,000 - $130,000 (average)",
    "market_trends": "Summary of current hiring demand and trends...",
    "top_companies": [
        "Company A",
        "Company B"
    ],
    "certifications": [
        "Certification X",
        "Certification Y"
    ],
    "project_ideas": [
        "Project Idea 1: Brief Description",
        "Project Idea 2: Brief Description"
    ],
    "interview_tips": [
        "Common topic or tip 1",
        "Common topic or tip 2"
    ],
    "learning_roadmap": "Brief outline of technologies to learn next..."
}}

Rules:
- Rely strictly on the search results for facts (like companies, certifications).
- Do not add markdown or extra explanations.
- Output ONLY valid JSON.

Search Results:
{results_text}
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "average_salary": "Not available",
            "market_trends": f"Could not retrieve live trends at this time: {str(e)}",
            "top_companies": [],
            "certifications": [],
            "project_ideas": [],
            "interview_tips": [],
            "learning_roadmap": "Not available"
        }

def get_career_why_and_what(job_role, candidate_skills):
    skills_str = ", ".join(candidate_skills)
    query = f"what is a {job_role} role? why would a candidate with skills {skills_str} be recommended for it?"
    try:
        search_results = tavily_client.search(query=query, search_depth="basic")
        results_text = "\n".join([_sanitize_text(r.get("content", "")) for r in search_results.get("results", [])])
    except Exception as e:
        results_text = f"Failed to fetch live search results: {str(e)}"
        
    prompt = f"""
    You are an expert career advisor.
    Based on the search results and the candidate's skills: {skills_str},
    explain "What" the job role '{job_role}' is, and "Why" this role is suggested for the candidate.
    
    Structure the response into a JSON object:
    {{
        "what": "Clear, concise definition of what the '{job_role}' role is and what they do.",
        "why": "Clear, customized explanation of why this role is suggested, linking candidate's skills specifically to the job expectations."
    }}
    
    Rules:
    - Keep both explanations brief (2-3 sentences each).
    - Be highly tailored to the skills: {skills_str}.
    - Output ONLY valid JSON.
    
    Search Results:
    {results_text}
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "what": f"A specialized technology role focused on {job_role}.",
            "why": f"This role matches your expertise in {', '.join(candidate_skills[:4])}."
        }

def get_active_jobs_and_internships(job_role):
    """Query Tavily for current job postings AND internship openings, then structure via Groq."""
    query = f"latest {job_role} job openings AND internship opportunities 2025 apply now site:linkedin.com OR site:naukri.com OR site:indeed.com OR site:glassdoor.com OR site:internshala.com"
    try:
        search_results = tavily_client.search(query=query, search_depth="basic")
        results_text = ""
        for r in search_results.get("results", []):
            title = _sanitize_text(r.get("title", ""))
            url = r.get("url", "")
            snippet = _sanitize_text(r.get("content", ""))
            results_text += f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n---\n"
    except Exception as e:
        results_text = f"Failed to fetch live search results: {str(e)}"

    prompt = f"""
You are an expert recruitment advisor.
Analyze the search results below and extract a list of 6-8 real, active job or internship opportunities for the role '{job_role}'.
Include a MIX of both full-time jobs AND internships.
Extract their title, company, source platform (e.g. LinkedIn, Naukri, Indeed, Internshala, Glassdoor), type ("Job" or "Internship"), the actual application URL, and a brief description.

Structure the response into a JSON object:
{{
    "listings": [
        {{
            "title": "Software Engineer Intern",
            "company": "Google",
            "platform": "LinkedIn",
            "type": "Internship",
            "url": "https://linkedin.com/jobs/...",
            "description": "Requires Python, Go, and good problem solving skills."
        }},
        {{
            "title": "Senior Data Analyst",
            "company": "Amazon",
            "platform": "Naukri",
            "type": "Job",
            "url": "https://naukri.com/...",
            "description": "5+ years experience in SQL, Python, and data visualization."
        }}
    ]
}}

Rules:
- Include ONLY opportunities that have real, valid, clickable HTTP/HTTPS links extracted from the search results.
- Do NOT fabricate or hallucinate URLs. If a listing has no valid URL in the search results, skip it.
- If you cannot find enough real listings, also include direct search links to LinkedIn Jobs, Naukri, Indeed, and Internshala for the role.
- Mark each listing as type "Job" or "Internship".
- Keep descriptions short (1-2 sentences).
- Use only ASCII characters in descriptions (no special currency symbols).
- Output ONLY valid JSON.

Search Results:
{results_text}
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        # Ensure listings exist
        if not result.get("listings"):
            result["listings"] = []
        return result
    except Exception as e:
        # Fallback: provide direct search links
        role_encoded = job_role.replace(' ', '%20')
        role_slug = job_role.lower().replace(' ', '-')
        return {
            "listings": [
                {
                    "title": f"{job_role} Jobs on LinkedIn",
                    "company": "Various Companies",
                    "platform": "LinkedIn",
                    "type": "Job",
                    "url": f"https://www.linkedin.com/jobs/search/?keywords={role_encoded}",
                    "description": "Browse active job listings on LinkedIn."
                },
                {
                    "title": f"{job_role} Jobs on Naukri",
                    "company": "Various Companies",
                    "platform": "Naukri.com",
                    "type": "Job",
                    "url": f"https://www.naukri.com/{role_slug}-jobs",
                    "description": "Browse active job listings on Naukri."
                },
                {
                    "title": f"{job_role} Internships on Internshala",
                    "company": "Various Companies",
                    "platform": "Internshala",
                    "type": "Internship",
                    "url": f"https://internshala.com/internships/{role_slug}-internship",
                    "description": "Browse active internship listings on Internshala."
                },
                {
                    "title": f"{job_role} on Indeed",
                    "company": "Various Companies",
                    "platform": "Indeed",
                    "type": "Job",
                    "url": f"https://www.indeed.com/jobs?q={role_encoded}",
                    "description": "Browse active listings on Indeed."
                }
            ]
        }
