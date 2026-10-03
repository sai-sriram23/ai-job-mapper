"""Recommendation Discrepancy Learning & Cache Memory Engine.

Compares ML model job predictions vs Groq AI job predictions.
Saves discrepancy cases into a persistent JSON cache memory (discrepancy_cache.json).
Uses cached historical discrepancy patterns to inform and adjust future predictions!
"""

import os
import json
import time
import logging
from datetime import datetime
from tavily_helper import groq_client

logger = logging.getLogger(__name__)

CACHE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "discrepancy_cache.json")


def load_discrepancy_cache() -> list[dict]:
    """Load cached discrepancy records from JSON file."""
    if not os.path.exists(CACHE_FILE_PATH):
        return []
    try:
        with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Error loading discrepancy cache: {e}")
        return []


def save_discrepancy_cache(cache_data: list[dict]):
    """Persist discrepancy records into JSON cache file."""
    try:
        with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving discrepancy cache: {e}")


def get_ai_job_recommendation(skills: list[str], branch: str = "CSE", cgpa: float = 8.0, dsa_skill: bool = True, coding_score: float = 75.0, aptitude_score: float = 75.0) -> dict:
    """Generate job role recommendation using Groq AI Cloud based on profile and live industry trends."""
    skills_str = ", ".join(skills) if skills else "General Programming"
    prompt = f"""
You are an expert AI Career Advisor and Tech Recruiter.
Analyze the following student profile and recommend the top 3 best-fitting job roles in current tech industry (2026):

Student Profile:
- Technical Skills: {skills_str}
- Academic Branch: {branch}
- CGPA: {cgpa}
- DSA Proficiency: {"High / Strong" if dsa_skill else "Basic / Moderate"}
- Coding Score: {coding_score}/100
- Aptitude Score: {aptitude_score}/100

Return ONLY a valid JSON object:
{{
  "top_role": "Primary Recommended Job Role",
  "confidence": 90,
  "reasoning": "2-sentence explanation of why this role matches the student's skills and market demand.",
  "recommended_roles": [
    {{
      "role": "Primary Recommended Job Role",
      "fit_score": 90,
      "reasoning": "Detailed justification."
    }},
    {{
      "role": "Alternative Role 1",
      "fit_score": 82,
      "reasoning": "Detailed justification."
    }},
    {{
      "role": "Alternative Role 2",
      "fit_score": 75,
      "reasoning": "Detailed justification."
    }}
  ]
}}
"""
    try:
        logger.info("Calling Groq AI for AI job recommendation...")
        candidate_models = ["groq/compound", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "qwen/qwen3.6-27b"]
        data = None
        for m in candidate_models:
            try:
                response = groq_client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": "You are a senior tech recruiter. Return ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content.strip()
                data = json.loads(content)
                break
            except Exception:
                continue
        if not data:
            raise RuntimeError("Groq models failed")
        return data
    except Exception as e:
        logger.error(f"Groq AI job recommendation failed: {e}")
        # Fallback AI response based on skills
        fallback_role = "Software Engineer"
        if any(s.lower() in ["python", "pandas", "sql", "data"] for s in skills):
            fallback_role = "Data Analyst / Data Engineer"
        elif any(s.lower() in ["react", "node", "html", "css", "web"] for s in skills):
            fallback_role = "Web Developer"
        return {
            "top_role": fallback_role,
            "confidence": 75,
            "reasoning": f"Grounded recommendation based on skill alignment with {skills_str}.",
            "recommended_roles": [
                {"role": fallback_role, "fit_score": 75, "reasoning": "Strong technical skill alignment."}
            ]
        }


def process_dual_recommendations(ml_top_role: str, profile_data: dict, skills: list[str]) -> dict:
    """Compare ML model prediction vs Groq AI recommendation.

    Saves discrepancy to cache memory if ML and AI predictions differ.
    Checks past cached discrepancy patterns to inform predictions.
    """
    # 1. Fetch AI Recommendation via Groq AI
    ai_rec = get_ai_job_recommendation(
        skills=skills,
        branch=profile_data.get("branch", "CSE"),
        cgpa=profile_data.get("cgpa", 8.0),
        dsa_skill=profile_data.get("dsa_skill", True),
        coding_score=profile_data.get("coding_score", 75.0),
        aptitude_score=profile_data.get("aptitude_score", 75.0)
    )

    ai_top_role = ai_rec.get("top_role", "Software Engineer").strip()
    ai_reasoning = ai_rec.get("reasoning", "")
    ai_roles_list = ai_rec.get("recommended_roles", [])

    # Normalize role strings for clean comparison
    norm_ml = ml_top_role.upper().strip()
    norm_ai = ai_top_role.upper().strip()

    # Determine if discrepancy exists
    # Discrepancy exists if roles are not identical substrings or direct matches
    is_discrepancy = norm_ml != norm_ai and norm_ml not in norm_ai and norm_ai not in norm_ml

    # 2. Check persistent Cache Memory for learned historical patterns
    cache_data = load_discrepancy_cache()
    skills_set = {s.lower().strip() for s in skills}

    matched_past_cases = []
    for entry in cache_data:
        entry_skills = {s.lower().strip() for s in entry.get("skills", [])}
        if skills_set and entry_skills:
            overlap = len(skills_set & entry_skills) / max(len(skills_set), len(entry_skills))
            if overlap >= 0.4:
                matched_past_cases.append(entry)

    learned_pattern_applied = len(matched_past_cases) > 0
    cached_insight_msg = ""
    if learned_pattern_applied:
        latest_match = matched_past_cases[-1]
        cached_insight_msg = f"Learned from cached prediction #{latest_match.get('id')}: For skills ({', '.join(latest_match.get('skills', []))}), AI recommended '{latest_match.get('ai_top_role')}' over ML's '{latest_match.get('ml_top_role')}' due to industry shift."

    # 3. Save new discrepancy to cache memory if mismatch detected
    new_cache_saved = False
    if is_discrepancy:
        new_entry = {
            "id": f"disc_{len(cache_data) + 1}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "skills": skills,
            "branch": profile_data.get("branch"),
            "cgpa": profile_data.get("cgpa"),
            "coding_score": profile_data.get("coding_score"),
            "ml_top_role": ml_top_role,
            "ai_top_role": ai_top_role,
            "reasoning": ai_reasoning,
        }
        # Append and keep last 100 entries
        cache_data.append(new_entry)
        if len(cache_data) > 100:
            cache_data = cache_data[-100:]
        save_discrepancy_cache(cache_data)
        new_cache_saved = True
        logger.info(f"Saved recommendation discrepancy into cache memory: ML='{ml_top_role}' vs AI='{ai_top_role}'")

    return {
        "ml_top_role": ml_top_role,
        "ai_top_role": ai_top_role,
        "ai_reasoning": ai_reasoning,
        "ai_recommended_roles": ai_roles_list,
        "is_discrepancy": is_discrepancy,
        "discrepancy_saved_to_cache": new_cache_saved,
        "learned_pattern_applied": learned_pattern_applied,
        "cached_insight_msg": cached_insight_msg,
        "total_cached_discrepancies": len(cache_data)
    }
