import os
import json
import logging
from tavily import TavilyClient
from groq import Groq

logger = logging.getLogger(__name__)

# Ensure environment variables are loaded
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"\'')

from api_key_manager import execute_groq_with_rotation, execute_tavily_with_rotation

class GroqProxy:
    """Proxy object that resolves Groq calls with dynamic multi-key rotation."""
    @property
    def chat(self):
        class ChatCompletions:
            def create(self, **kwargs):
                prompt = ""
                messages = kwargs.get("messages", [])
                if messages:
                    prompt = messages[-1].get("content", "")
                temp = kwargs.get("temperature", 0.3)
                is_json = kwargs.get("response_format", {}).get("type") == "json_object"
                max_tok = kwargs.get("max_tokens", 4096)
                content = execute_groq_with_rotation(prompt, temperature=temp, is_json=is_json, max_tokens=max_tok)
                
                class Choice:
                    def __init__(self, text):
                        self.message = type("Msg", (), {"content": text})()
                return type("Response", (), {"choices": [Choice(content)]})()
        class Chat:
            completions = ChatCompletions()
        return Chat()

class TavilyProxy:
    """Proxy object that resolves Tavily search with multi-key rotation."""
    def search(self, query: str, **kwargs):
        depth = kwargs.get("search_depth", "basic")
        return execute_tavily_with_rotation(lambda client: client.search(query=query, search_depth=depth))

groq_client = GroqProxy()
tavily_client = TavilyProxy()

def _sanitize_text(text):
    """Remove non-ASCII characters that crash Windows charmap codec."""
    if not text:
        return ""
    return text.encode("ascii", errors="ignore").decode("ascii")


def parse_insights_json(text):
    """Robust JSON parser that ensures a dictionary is returned."""
    text = text.strip()
    
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        
    data = json.loads(text)
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("Decoded JSON is not a dictionary object.")
    return data

SUB_TO_PARENT_ROLE = {
    'ACCESSIBILITY SPECIALIST': 'Software Engineer / Web Developer',
    'AGILE PROJECT MANAGER': 'Project Manager / IT Analyst',
    'BUSINESS SYSTEMS ANALYST': 'Business Analyst / Systems Analyst',
    'CLOUD ARCHITECT': 'Cloud Engineer / Software Engineer',
    'COMPUTER GRAPHICS ANIMATOR': 'Web Developer / UI Designer',
    'DATA ANALYST': 'Data Analyst / Business Intelligence',
    'DATA MODELER': 'Data Engineer / Data Scientist',
    'DATA SCIENTIST': 'Data Scientist / Machine Learning Engineer',
    'DEVOPS MANAGER': 'DevOps Engineer / Software Engineer',
    'FRAMEWORKS SPECIALIST': 'Software Engineer / Web Developer',
    'INFORMATION ARCHITECT': 'UX Designer / Analyst',
    'INTERACTION DESIGNER': 'UI/UX Designer / Web Developer',
    'MOBILE APP DEVELOPER': 'Software Engineer / Mobile Developer',
    'PRODUCT MANAGER': 'Product Manager / Business Analyst',
    'SECURITY SPECIALIST': 'Cybersecurity Engineer / Software Engineer',
    'TECHNICAL ACCOUNT MANAGER': 'IT Specialist / Customer Success Engineer',
    'TECHNICAL LEAD': 'Software Engineer / Technical Lead'
}


def search_job_market(query, feature: str = "market"):
    return execute_tavily_with_rotation(lambda client: client.search(query=query, search_depth="basic"), feature=feature)


def call_groq_with_fallback(prompt: str, temperature: float = 0.3, is_json: bool = True, feature: str = None) -> str:
    """Execute Groq API call with automatic multi-key rotation and feature-targeted key routing."""
    return execute_groq_with_rotation(prompt=prompt, temperature=temperature, is_json=is_json, feature=feature)



def get_job_market_insights(job_role: str, user_skills: list = None) -> dict:
    """Fetch live web data via Tavily and generate enhanced 2026 market insights using LangChain engine."""
    try:
        from langchain_engine import get_enhanced_market_insights_langchain
        return get_enhanced_market_insights_langchain(job_role, user_skills)
    except Exception as e:
        logger.warning(f"LangChain market insights delegate error ({e}), falling back to direct Tavily search...")
        query = f"current 2026 job market trends, average salary, top hiring companies for '{job_role}'"
        tavily_data = search_job_market(query, feature="market")
        
        results = tavily_data.get("results", [])
        results_text = "\n".join([f"- {r.get('title')}: {_sanitize_text(r.get('content'))}" for r in results[:5]])
        clean_results = str(results_text).replace('"', ' ')

        prompt = f"Analyze search results for '{job_role}'.\n" \
                 "Return JSON with keys: average_salary, market_trends, top_companies, key_demanded_skills, sources.\n\n" \
                 "Search Results:\n" + clean_results

        try:
            raw_resp = call_groq_with_fallback(prompt, temperature=0.3, is_json=True, feature="market")
            return parse_insights_json(raw_resp)
        except Exception:
            return {
                "average_salary": f"Competitive industry rate for {job_role}",
                "market_trends": f"High demand for specialized skills in {job_role}.",
                "top_companies": [f"Tech Leaders hiring {job_role}"],
                "key_demanded_skills": ["Problem Solving", "Domain Expertise"],
                "sources": ["2026 Industry Reports"]
            }



def get_career_why_and_what(job_role: str, candidate_skills) -> dict:
    """Generate 'Why this role?' and 'What to learn next?' insights using Groq AI."""
    skills_str = ", ".join(candidate_skills) if isinstance(candidate_skills, list) else str(candidate_skills)

    query = f"what does a {job_role} do? why is a candidate with {skills_str} a good fit for {job_role}?"
    tavily_data = search_job_market(query, feature="market")
    results = tavily_data.get("results", [])
    results_text = "\n".join([f"- {r.get('title')}: {_sanitize_text(r.get('content'))}" for r in results[:3]])
    clean_results = str(results_text).replace('"', ' ')

    prompt = f"Candidate skills: {skills_str}. Target role: {job_role}.\n" \
             "Explain What the role does and Why candidate skills qualify them.\n" \
             "Return JSON with keys 'what' and 'why'.\n\nSearch Results:\n" + clean_results

    try:
        raw_resp = call_groq_with_fallback(prompt, temperature=0.3, is_json=True, feature="market")
        return parse_insights_json(raw_resp)
    except Exception as e:
        return {
            "what": f"A {job_role} is responsible for designing, deploying, and maintaining specialized solutions for the organization.",
            "why": f"Your background in {skills_str[:30]} provides a strong foundation for the day-to-day requirements of a {job_role}."
        }


def get_active_jobs_and_internships(job_role: str, location: str = "All", job_type: str = "All") -> dict:
    """Query Tavily for current job postings AND internship openings, then structure via Groq."""
    parent_role = SUB_TO_PARENT_ROLE.get(job_role.upper())
    loc_clause = f" in {location}" if location and location != "All" else ""
    type_clause = f" {job_type}" if job_type and job_type != "All" else " job openings AND internship opportunities"

    if parent_role:
        query = f"latest {job_role} ({parent_role}){type_clause}{loc_clause} 2025 apply now site:linkedin.com OR site:naukri.com OR site:indeed.com OR site:internshala.com OR site:wellfound.com"
    else:
        query = f"latest {job_role}{type_clause}{loc_clause} 2025 apply now site:linkedin.com OR site:naukri.com OR site:indeed.com OR site:internshala.com OR site:wellfound.com"
        
    tavily_data = search_job_market(query, feature="jobsearch")
    results = tavily_data.get("results", [])
    results_text = ""
    for r in results[:6]:
        title = _sanitize_text(r.get("title", ""))
        url = r.get("url", "")
        snippet = _sanitize_text(r.get("content", ""))
        results_text += f"Title: {title} | URL: {url} | Snippet: {snippet}\n"

    clean_results = str(results_text).replace('"', ' ')
    prompt = f"Analyze search results for {job_role}{loc_clause}.\n" \
             "Extract 6-8 active jobs or internships with title, company, platform, type, url, description.\n" \
             "Return JSON with key 'listings'.\n\nSearch Results:\n" + clean_results

    try:
        raw_resp = call_groq_with_fallback(prompt, temperature=0.1, is_json=True, feature="jobsearch")
        result = parse_insights_json(raw_resp)
        if not result.get("listings"):
            result["listings"] = []
        return result
    except Exception as e:
        role_encoded = job_role.replace(' ', '%20')
        role_slug = job_role.lower().replace(' ', '-')
        loc_param = f"&location={location}" if location != "All" else ""
        return {
            "listings": [
                {
                    "title": f"{job_role} Positions on LinkedIn",
                    "company": "Hiring Partners",
                    "platform": "LinkedIn",
                    "type": job_type if job_type != "All" else "Job",
                    "url": f"https://www.linkedin.com/jobs/search/?keywords={role_encoded}{loc_param}",
                    "description": f"Browse real-time {job_role} job and internship openings."
                },
                {
                    "title": f"{job_role} Opportunities on Naukri",
                    "company": "Top Employers",
                    "platform": "Naukri.com",
                    "type": job_type if job_type != "All" else "Job",
                    "url": f"https://www.naukri.com/{role_slug}-jobs",
                    "description": f"Explore latest verified hiring listings on Naukri."
                },
                {
                    "title": f"{job_role} Internships on Internshala",
                    "company": "Growth Startups",
                    "platform": "Internshala",
                    "type": "Internship",
                    "url": f"https://internshala.com/internships/{role_slug}-internship",
                    "description": "Find top paid internship programs matching your profile."
                }
            ]
        }

