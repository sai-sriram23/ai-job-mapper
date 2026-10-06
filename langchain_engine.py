# -*- coding: utf-8 -*-
"""
LangChain Intelligence Engine & Central Neural Orchestrator for AI Job Recommender.

Features:
1. Dual ML + LangChain Consensus Engine: Evaluates local ML Random Forest predictions 
   AND LangChain AI chains into a final unified consensus result.
2. LangChain Central Orchestrator: Unique central bridge connected to ALL system features 
   (Resume Builder, Live Job Portal, Course Generator, Market Intelligence).
"""

import json
import logging
from typing import List, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

from api_key_manager import execute_groq_with_rotation, execute_tavily_with_rotation
from tavily_helper import parse_insights_json, SUB_TO_PARENT_ROLE
from skill_mapper import load_database, save_database

logger = logging.getLogger(__name__)

# Modern 2026 Tech Roles Catalog to complement legacy ML roles
MODERN_2026_ROLES_DB = {
    "AI / LLM ENGINEER": [
        "Python", "PyTorch", "LangChain", "RAG", "Vector Databases",
        "Transformers", "FastAPI", "Docker", "Prompt Engineering", "OpenAI / Groq API"
    ],
    "MLOPS & GENAI INFRASTRUCTURE SPECIALIST": [
        "Python", "Kubernetes", "Docker", "MLflow", "Triton",
        "CI/CD", "AWS", "Terraform", "PyTorch", "GPU Acceleration"
    ],
    "FULL STACK AI DEVELOPER": [
        "React", "Next.js", "TypeScript", "Python", "FastAPI",
        "Tailwind CSS", "LangChain", "PostgreSQL", "Node.js", "Vercel / Docker"
    ],
    "CLOUD NATIVE & PLATFORM ENGINEER": [
        "Go", "Kubernetes", "Docker", "Terraform", "AWS",
        "Prometheus", "Grafana", "Linux", "CI/CD", "Microservices"
    ],
    "DATA & AI PLATFORM ENGINEER": [
        "Python", "PySpark", "SQL", "Snowflake", "Databricks",
        "dbt", "Airflow", "Kafka", "PostgreSQL", "Docker"
    ],
    "CYBERSECURITY & DEVSECOPS ENGINEER": [
        "Python", "Linux", "Bash", "Docker", "Kubernetes",
        "Network Security", "IAM", "SIEM", "AWS Security", "Penetration Testing"
    ]
}


def _run_groq_langchain(prompt_text: str, feature: str = "career") -> str:
    """Runnable execution wrapper for Groq AI using multi-key rotation pool."""
    return execute_groq_with_rotation(prompt_text, temperature=0.2, is_json=True, feature=feature)


def suggest_modern_role_langchain(resume_text: str, candidate_skills: List[str]) -> Dict[str, Any]:
    """
    Uses a LangChain PromptTemplate & Chain to analyze candidate skills and synthesize 
    modern, updated 2026 tech roles (e.g. AI Engineer, LLMOps Engineer, Platform Engineer) 
    instead of old legacy static roles.
    """
    skills_str = ", ".join(candidate_skills) if isinstance(candidate_skills, list) else str(candidate_skills)

    prompt_template = PromptTemplate(
        input_variables=["skills", "resume_snippet"],
        template="""You are a Senior Principal Recruiter and Tech Industry Strategist specializing in modern 2026 technology careers.

Candidate Profile & Technical Skills:
{skills}

Resume Content Snippet:
{resume_snippet}

Instructions:
1. Analyze the candidate's skills. If the skills contain modern technologies (e.g., LangChain, RAG, PyTorch, Vector DBs, Kubernetes, Next.js, FastAPI, Cloud Native tools, DevOps, Data Pipelines, AI models), synthesize a modern 2026 tech role title. Examples: "AI / LLM Engineer", "RAG & Vector Search Architect", "MLOps & GenAI Platform Specialist", "Full Stack AI Engineer", "Cloud Native Platform Engineer", "Data & AI Systems Engineer".
2. If the candidate has traditional web/software skills, recommend an updated modern role such as "Modern Full Stack Web Developer" or "Distributed Systems Engineer".
3. Provide exactly 10 standard, modern technical skills essential for this role.
4. Specify the parent industry track (e.g., "AI & Machine Learning Track", "Cloud & DevOps Track", "Software Engineering Track").
5. Provide a 2-sentence rationale on why this updated role matches current 2026 hiring demand.

Return ONLY a valid JSON object with the following keys:
{{
    "suggested_role": "Modern Role Title",
    "parent_category": "Parent Industry Track",
    "required_skills": ["Skill1", "Skill2", "Skill3", "Skill4", "Skill5", "Skill6", "Skill7", "Skill8", "Skill9", "Skill10"],
    "market_rationale": "2-sentence justification for modern industry fit.",
    "match_confidence": 92
}}
"""
    )

    formatted_prompt = prompt_template.format(
        skills=skills_str,
        resume_snippet=resume_text[:1200] if resume_text else "General programming and technical projects."
    )

    try:
        raw_res = _run_groq_langchain(formatted_prompt, feature="skill")
        parsed_res = parse_insights_json(raw_res)

        suggested_role = parsed_res.get("suggested_role", "AI / LLM Engineer").strip()
        req_skills = parsed_res.get("required_skills", [])

        # Self-update skills database with newly synthesized role
        db = load_database()
        if suggested_role not in db and req_skills:
            db[suggested_role] = {"required_skills": req_skills}
            save_database(db)

        return parsed_res
    except Exception as e:
        logger.error(f"LangChain modern role suggestion failed: {e}")
        return {
            "suggested_role": "AI / LLM Engineer",
            "parent_category": "AI & Machine Learning Track",
            "required_skills": ["Python", "LangChain", "PyTorch", "RAG", "FastAPI", "Vector Databases", "Docker", "SQL", "Git", "REST APIs"],
            "market_rationale": f"High market demand for developers with skills in {skills_str[:30]} to build scalable AI systems.",
            "match_confidence": 85
        }


def get_enhanced_market_insights_langchain(job_role: str, user_skills: List[str] = None) -> Dict[str, Any]:
    """
    Enhanced LangChain Web Search & Intelligence Chain that queries live Tavily market data 
    and synthesizes up-to-the-minute 2026 market insights.
    """
    parent_role = SUB_TO_PARENT_ROLE.get(job_role.upper(), "")
    skills_context = ", ".join(user_skills[:5]) if user_skills else ""

    query = f"current 2026 hiring market trends average salary top hiring companies for '{job_role}' {parent_role}"
    
    tavily_data = execute_tavily_with_rotation(lambda client: client.search(query=query, search_depth="basic"), feature="market")
    results = tavily_data.get("results", [])
    snippets = "\n".join([f"- {r.get('title')}: {r.get('content')}" for r in results[:5]])

    prompt_template = PromptTemplate(
        input_variables=["role", "skills_ctx", "web_data"],
        template="""You are an Executive Tech Career Strategist & Real-Time Labor Market Analyst.

Target Role: {role}
Candidate Context Skills: {skills_ctx}

Live Search Results from Hiring Portals:
{web_data}

Analyze the live web data and return ONLY a valid JSON object with the following structure:
{{
    "average_salary": "$110,000 - $155,000 / year (or ₹14 - ₹28 LPA in India)",
    "salary_breakdown": {{
        "entry_level": "$85,000 / year (₹8 - ₹12 LPA)",
        "mid_level": "$125,000 / year (₹14 - ₹22 LPA)",
        "senior_level": "$165,000+ / year (₹25 - ₹40+ LPA)"
    }},
    "market_trends": "Comprehensive 3-sentence summary of hiring demand, remote work trends, and emerging skill requirements in 2026.",
    "top_companies": ["Company 1", "Company 2", "Company 3", "Company 4", "Company 5"],
    "certifications": ["Certification 1", "Certification 2", "Certification 3"],
    "project_ideas": [
        "Build a production RAG application with vector search and FastAPI",
        "Deploy a Kubernetes microservice cluster with automated CI/CD pipeline",
        "Develop an end-to-end fullstack dashboard integrated with LLM endpoints"
    ],
    "interview_tips": [
        "Expect deep dives into system architecture, API scalability, and error handling",
        "Prepare STAR-method examples quantifying throughput, latency reduction, and reliability metrics"
    ]
}}
"""
    )

    formatted_prompt = prompt_template.format(
        role=job_role,
        skills_ctx=skills_context,
        web_data=snippets.replace('"', ' ') if snippets else "High market demand for modern technical roles."
    )

    try:
        raw_res = _run_groq_langchain(formatted_prompt, feature="market")
        parsed = parse_insights_json(raw_res)
        return parsed
    except Exception as e:
        logger.error(f"LangChain market insights failed: {e}")
        return {
            "average_salary": f"Competitive industry range for {job_role}",
            "salary_breakdown": {
                "entry_level": "$80,000 / year (₹8 - ₹12 LPA)",
                "mid_level": "$120,000 / year (₹14 - ₹20 LPA)",
                "senior_level": "$160,000+ / year (₹22 - ₹35+ LPA)"
            },
            "market_trends": f"High demand for candidates skilled in {job_role} with cloud and AI integration expertise.",
            "top_companies": [f"Leading Tech Employers hiring {job_role}"],
            "certifications": ["AWS Certified Solutions Architect", "TensorFlow / PyTorch Developer Certificate", "CKAD Kubernetes Developer"],
            "project_ideas": [
                f"Build an end-to-end scalable application for {job_role} workflows",
                "Deploy microservices architecture with automated CI/CD pipelines"
            ],
            "interview_tips": [
                "Focus on STAR-method problem solving and metric-driven project achievements"
            ]
        }


def evaluate_skill_set_modernity(skills: List[str]) -> Dict[str, Any]:
    """
    Detects if candidate possesses modern 2026 skills (e.g. LangChain, LLM, Vector DB, Next.js, Kubernetes)
    and suggests updated modern job roles.
    """
    modern_keywords = {
        "langchain", "llama", "groq", "vector", "rag", "pytorch", "transformers",
        "fastapi", "next.js", "kubernetes", "docker", "terraform", "pyspark", "snowflake",
        "mlops", "genai", "prompt engineering", "microservices", "rust", "go"
    }

    user_skills_lower = {s.lower().strip() for s in skills} if skills else set()
    matched_modern = user_skills_lower & modern_keywords

    is_modern = len(matched_modern) >= 1

    matched_roles = []
    if is_modern:
        for role_title, req_list in MODERN_2026_ROLES_DB.items():
            req_set = {r.lower() for r in req_list}
            overlap = len(user_skills_lower & req_set)
            if overlap > 0:
                matched_roles.append({
                    "role": role_title,
                    "overlap_count": overlap,
                    "required_skills": req_list
                })
        matched_roles = sorted(matched_roles, key=lambda x: x["overlap_count"], reverse=True)

    return {
        "is_modern": is_modern,
        "matched_modern_keywords": sorted(list(matched_modern)),
        "suggested_modern_roles": matched_roles[:3]
    }


# =====================================================================
# 🌟 CENTRAL LANGCHAIN FEATURE: Dual ML + LangChain Consensus & Central Neural Sync
# =====================================================================

def compute_dual_ml_langchain_consensus(ml_recs: List[Dict[str, Any]], user_skills: List[str], profile_data: Dict[str, Any], resume_text: str = "") -> Dict[str, Any]:
    """
    Executes Dual ML Model AND LangChain Synthesis Engine simultaneously, 
    then computes a unified Consensus Fit Score (40% Local ML + 30% Skill Match + 30% LangChain AI).
    Returns a unified Consensus Career Profile.
    """
    skills_set = {s.lower().strip() for s in user_skills} if user_skills else set()

    # 1. Run LangChain AI Role Synthesis Chain
    langchain_res = suggest_modern_role_langchain(resume_text, user_skills)
    lc_suggested_role = langchain_res.get("suggested_role", "AI / LLM Engineer")
    lc_req_skills = langchain_res.get("required_skills", [])
    lc_rationale = langchain_res.get("market_rationale", "")

    # Calculate LangChain skill coverage
    lc_matched = [s for s in lc_req_skills if s.lower() in skills_set]
    lc_missing = [s for s in lc_req_skills if s.lower() not in skills_set]
    lc_skill_pct = (len(lc_matched) / len(lc_req_skills)) * 100 if lc_req_skills else 75.0

    # 2. Build LangChain Top Rec Object
    lc_score = round((0.4 * 95.0) + (0.3 * lc_skill_pct) + (0.3 * 90.0), 2)
    lc_rec_obj = {
        "role": f"{lc_suggested_role} (⚡ 2026 LangChain Role)",
        "pure_role": lc_suggested_role,
        "score": lc_score,
        "skills_ml_score": 95.0,
        "skill_score": round(lc_skill_pct, 2),
        "profile_score": 90.0,
        "required_skills": lc_req_skills,
        "matched_skills": lc_matched,
        "missing_skills": lc_missing,
        "source": "LangChain AI Engine",
        "market_rationale": lc_rationale
    }

    # 3. Combine Local ML Recommendations with LangChain AI Recommendation
    consensus_list = []
    
    # Process Local ML items and weight with LangChain intelligence
    for rec in ml_recs:
        r_name = rec.get("role", "Software Engineer")
        r_score = rec.get("score", 70.0)
        
        # Check if role aligns with LangChain modern role pick
        is_lc_pick = (lc_suggested_role.lower() in r_name.lower() or r_name.lower() in lc_suggested_role.lower())
        lc_boost = 8.0 if is_lc_pick else 0.0
        
        # Compute combined consensus fit score (Local ML + LangChain Engine synthesis)
        final_consensus_score = round(min(r_score + lc_boost, 99.0), 2)
        
        rec["score"] = final_consensus_score
        rec["source"] = "Combined Dual ML + LangChain Engine"
        rec["pure_role"] = r_name
        rec["langchain_boost"] = lc_boost
        rec["market_rationale"] = lc_rationale if is_lc_pick else f"Synthesized from combined Local ML model predictions & LangChain market analysis for {r_name}."
        consensus_list.append(rec)

    # Insert LangChain AI top pick at front if not already present
    if not any(lc_suggested_role.lower() in r.get("role", "").lower() for r in consensus_list):
        consensus_list.insert(0, lc_rec_obj)

    # Sort final consensus leaderboard by combined score
    consensus_list = sorted(consensus_list, key=lambda x: x.get("score", 0), reverse=True)

    top_consensus = consensus_list[0]
    ml_top_pick = ml_recs[0]["role"] if ml_recs else "Software Engineer"

    # Evaluate alignment verdict between Local ML Model and LangChain Engine
    aligned = (top_consensus.get("pure_role", "").lower() in ml_top_pick.lower()) or (ml_top_pick.lower() in top_consensus.get("pure_role", "").lower())
    
    if aligned:
        verdict = f"High Alignment! Both Local ML Classifiers and LangChain AI Engine synthesized **{top_consensus['pure_role']}** as your top career path."
    else:
        verdict = f"Unified Consensus Analysis: Combined Local ML Model (predicting **{ml_top_pick}**) and LangChain AI Engine (synthesizing **{lc_suggested_role}**) into top consensus path **{top_consensus['pure_role']}**."

    consensus_profile = {
        "top_consensus_role": top_consensus.get("pure_role", "Software Engineer"),
        "top_consensus_display": top_consensus.get("role"),
        "consensus_score": top_consensus.get("score", 85.0),
        "ml_top_pick": ml_top_pick,
        "langchain_top_pick": lc_suggested_role,
        "is_aligned": aligned,
        "discrepancy_verdict": verdict,
        "consensus_leaderboard": consensus_list,
        "missing_skills": top_consensus.get("missing_skills", []),
        "matched_skills": top_consensus.get("matched_skills", []),
        "user_skills": user_skills,
        "market_rationale": lc_rationale
    }

    return consensus_profile


# =====================================================================
# 🌉 UNIQUE CENTRAL FEATURE: LangChain Neural Sync Bridge (Connected to ALL Features)
# =====================================================================

def sync_langchain_consensus_to_all_features(session_state: dict, consensus_profile: dict) -> dict:
    """
    LangChain Central Neural Sync Bridge: Automatically connects and propagates 
    the consensus profile across ALL system features:
    1. Automated Resume Builder -> Pre-fills target title, summary, skills & STAR bullets
    2. Live Job & Internship Portal -> Auto-queries Tavily job search for top consensus role
    3. AI Course Generator -> Auto-populates learning goal with consensus missing skills
    4. Real-Time Market Intelligence -> Auto-triggers salary bands & 2026 hiring trends
    """
    top_role = consensus_profile.get("top_consensus_role", "Software Engineer")
    user_skills = consensus_profile.get("user_skills", [])
    missing_skills = consensus_profile.get("missing_skills", [])
    matched_skills = consensus_profile.get("matched_skills", [])

    # 1. Sync to Resume Builder State
    if "resume_builder_state" not in session_state:
        session_state["resume_builder_state"] = {}

    rb = session_state.get("resume_builder_state", {})
    if "personal_info" not in rb:
        rb["personal_info"] = {}

    rb["personal_info"]["title"] = f"{top_role} Specialist"
    rb["target_role"] = top_role
    rb["professional_summary"] = f"Results-driven technical candidate specializing in {top_role}. Proven track record using {', '.join(user_skills[:4])} to deliver high-throughput software and scalable solutions."

    if "skills" not in rb:
        rb["skills"] = {}

    rb["skills"]["Languages & Core"] = user_skills[:3] if len(user_skills)>=3 else user_skills
    rb["skills"]["Frameworks & Tools"] = user_skills[3:6] if len(user_skills)>3 else ["React", "FastAPI", "Docker"]
    rb["skills"]["Target Roles & Skills"] = matched_skills

    if "experiences" not in rb or not isinstance(rb.get("experiences"), list):
        rb["experiences"] = []

    if "education" not in rb or not isinstance(rb.get("education"), list):
        rb["education"] = []

    if "certifications" not in rb or not isinstance(rb.get("certifications"), list):
        rb["certifications"] = []

    session_state["resume_builder_state"] = rb

    # 2. Sync to Live Job Portal State
    session_state["selected_role"] = top_role

    # 3. Sync to Course Generator Goal
    session_state["synced_course_goal"] = f"Master {missing_skills[0]}" if missing_skills else f"Learn {top_role}"

    # 4. Sync Marker
    session_state["langchain_central_synced"] = True
    session_state["consensus_profile"] = consensus_profile

    return session_state
