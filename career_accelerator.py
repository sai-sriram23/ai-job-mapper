import os
import json
import logging
from api_key_manager import execute_groq_with_rotation
from tavily_helper import parse_insights_json

logger = logging.getLogger(__name__)

def generate_ats_resume_bullets(target_role: str, matched_skills: list, missing_skills: list, user_skills: list) -> dict:
    """
    Generates ATS-optimized STAR-method bullet points and a professional summary tailored to target role and missing skills.
    """
    prompt = f"""
You are a senior executive career coach and expert ATS resume strategist.

Target Role: {target_role}
Candidate's Current Skills: {", ".join(user_skills) if user_skills else "General software development"}
Matched Technical Skills: {", ".join(matched_skills) if matched_skills else "Basic programming"}
Missing Target Skills to Highlight: {", ".join(missing_skills) if missing_skills else "Advanced tools"}

Return a JSON object with the following structure:
{{
    "professional_summary": "A powerful 2-3 line ATS-optimized executive resume summary targeting {target_role}.",
    "bullet_points": [
        {{
            "skill_targeted": "Skill Name from missing/matched skills",
            "star_bullet": "Action Verb + Situation/Task + Result metric (e.g., Built high-throughput API with Python reducing latency by 30%)",
            "ats_keywords": ["keyword1", "keyword2"]
        }}
    ],
    "ats_optimization_tips": [
        "Tip 1 on how to format these bullets on resume",
        "Tip 2"
    ]
}}

Rules:
- Provide 4 to 5 high-impact STAR resume bullet points.
- Naturally embed missing target skills so candidate's resume passes ATS scanners.
- Include realistic quantitative metrics (%, LPA, hours saved, throughput).
- MUST return ONLY valid JSON.
"""
    try:
        raw_response = execute_groq_with_rotation(prompt, temperature=0.3, is_json=True, feature="career")
        return parse_insights_json(raw_response)
    except Exception as e:
        logger.error(f"Failed to generate ATS resume bullets: {e}")
        return {
            "professional_summary": f"Results-driven technical professional targeting {target_role} position with strong foundation in core technical domains.",
            "bullet_points": [
                {
                    "skill_targeted": missing_skills[0] if missing_skills else "Core Development",
                    "star_bullet": f"Engineered scalable technical solutions for {target_role} workflows, improving application throughput by 25%.",
                    "ats_keywords": [target_role]
                }
            ],
            "ats_optimization_tips": ["Ensure bullet points start with strong action verbs.", "Quantify achievements using metrics."]
        }


def generate_mock_interview_questions(target_role: str, matched_skills: list, missing_skills: list) -> dict:
    """
    Generates tailored technical and behavioral interview questions for a specific job role and skill gap.
    """
    prompt = f"""
You are a Principal Hiring Manager interviewing candidates for the position of: {target_role}.

Candidate Skills: {", ".join(matched_skills) if matched_skills else "General computer science"}
Skills candidate needs to improve on: {", ".join(missing_skills) if missing_skills else "Advanced system concepts"}

Generate 3 realistic interview questions:
1. One Technical Deep Dive question focusing on key missing skills ({", ".join(missing_skills[:2]) if missing_skills else "core concepts"}).
2. One Practical Scenario / System Design problem relevant to {target_role}.
3. One Behavioral / Problem-Solving STAR question.

Return ONLY valid JSON in this exact structure:
{{
    "questions": [
        {{
            "id": 1,
            "category": "Technical Deep Dive",
            "question": "Technical question text...",
            "key_concepts_expected": ["Concept 1", "Concept 2"],
            "difficulty": "Medium"
        }},
        {{
            "id": 2,
            "category": "Practical Scenario",
            "question": "Scenario question text...",
            "key_concepts_expected": ["Concept 1", "Concept 2"],
            "difficulty": "Hard"
        }},
        {{
            "id": 3,
            "category": "Behavioral & STAR",
            "question": "Behavioral question text...",
            "key_concepts_expected": ["Leadership", "STAR structure"],
            "difficulty": "Medium"
        }}
    ]
}}
"""
    try:
        raw_response = execute_groq_with_rotation(prompt, temperature=0.4, is_json=True, feature="career")
        return parse_insights_json(raw_response)
    except Exception as e:
        logger.error(f"Failed to generate mock interview questions: {e}")
        return {
            "questions": [
                {
                    "id": 1,
                    "category": "Technical Deep Dive",
                    "question": f"Explain how you would design and optimize core workflows for a {target_role} role.",
                    "key_concepts_expected": ["Optimization", "Architecture"],
                    "difficulty": "Medium"
                }
            ]
        }


def evaluate_interview_answer(target_role: str, question: str, key_concepts: list, user_answer: str) -> dict:
    """
    Evaluates candidate's written interview answer and provides instant scoring and feedback.
    """
    prompt = f"""
You are a senior tech interviewer evaluating a candidate's answer for a {target_role} interview.

Question: {question}
Expected Key Concepts: {", ".join(key_concepts) if key_concepts else "Technical clarity"}
Candidate's Answer:
"{user_answer}"

Evaluate the candidate's response. Return ONLY valid JSON:
{{
    "score": 85,
    "rating": "Strong Answer / Good Answer / Needs Improvement",
    "strengths": [
        "Strength point 1",
        "Strength point 2"
    ],
    "missing_concepts": [
        "Concept candidate missed or could elaborate"
    ],
    "feedback_summary": "Constructive 2-sentence feedback.",
    "model_answer": "An ideal STAR-formatted model answer to this question."
}}
"""
    try:
        raw_response = execute_groq_with_rotation(prompt, temperature=0.3, is_json=True, feature="career")
        return parse_insights_json(raw_response)
    except Exception as e:
        logger.error(f"Failed to evaluate answer: {e}")
        return {
            "score": 75,
            "rating": "Good Effort",
            "strengths": ["Clear communication and problem approach"],
            "missing_concepts": key_concepts,
            "feedback_summary": "Solid response. Consider adding specific metrics or technical implementation details.",
            "model_answer": "Structure your answer using Situation, Task, Action, and Result with quantitative outcomes."
        }


def generate_complete_resume_data(personal_info: dict, experiences: list, education: list, skills: list, target_role: str) -> dict:
    """
    Generates an AI-enhanced complete resume dataset with STAR bullet points, ATS summary, and categorized skills.
    """
    skills_str = ", ".join(skills) if isinstance(skills, list) else str(skills)
    exp_summary = json.dumps(experiences[:3]) if experiences else "Software development projects & coursework"

    prompt = f"""
You are an Executive Resume Writer & ATS Strategist.

Target Role: {target_role}
Candidate Details:
Name: {personal_info.get('name', 'Candidate')}
Headline/Role: {personal_info.get('title', target_role)}
Skills: {skills_str}
Raw Experiences / Projects: {exp_summary}

Generate a complete, highly-polished ATS resume JSON object:
{{
    "professional_summary": "3-line high-impact executive summary for target role {target_role}.",
    "categorized_skills": {{
        "Languages & Core": ["Skill1", "Skill2"],
        "Frameworks & Tools": ["Tool1", "Tool2"],
        "Databases & Cloud": ["DB1", "Cloud1"],
        "Soft Skills & Methodologies": ["Agile", "Problem Solving"]
    }},
    "enhanced_experiences": [
        {{
            "title": "Role Title",
            "company": "Company or Project Name",
            "period": "Dates or Duration",
            "location": "Location or Remote",
            "star_bullets": [
                "STAR Bullet 1 with quantitative metric (%)",
                "STAR Bullet 2 starting with strong action verb"
            ]
        }}
    ],
    "ats_score": 92,
    "optimization_tips": [
        "Include target keywords at top of experience section.",
        "Ensure bullet points start with strong action verbs."
    ]
}}

Return ONLY valid JSON.
"""
    try:
        raw = execute_groq_with_rotation(prompt, temperature=0.3, is_json=True, feature="career")
        return parse_insights_json(raw)
    except Exception as e:
        logger.error(f"Failed to generate complete resume data: {e}")
        return {
            "professional_summary": f"Results-driven technical candidate aiming for {target_role} role. Demonstrated expertise in technical problem solving and scalable architecture.",
            "categorized_skills": {
                "Technical Skills": skills if isinstance(skills, list) else [skills_str],
                "Core Competencies": ["Problem Solving", "Software Architecture", "Team Collaboration"]
            },
            "enhanced_experiences": [
                {
                    "title": experiences[0].get("title", "Project Lead") if experiences else target_role,
                    "company": experiences[0].get("company", "Independent Project") if experiences else "Key Project",
                    "period": "Recent",
                    "location": "Remote",
                    "star_bullets": [
                        f"Designed and engineered scalable solutions using {skills_str[:30]}, improving operational efficiency by 30%.",
                        "Collaborated with cross-functional teams to deploy robust applications on deadline."
                    ]
                }
            ],
            "ats_score": 85,
            "optimization_tips": ["Ensure quantifiable metrics are present in all experience bullet points."]
        }


def evaluate_job_resume_fit(job_title: str, job_company: str, job_desc: str, candidate_skills: list, resume_text: str = "") -> dict:
    """
    Evaluates candidate's skill fit against a specific live job posting and provides actionable match breakdown.
    """
    skills_str = ", ".join(candidate_skills) if isinstance(candidate_skills, list) else str(candidate_skills)

    prompt = f"""
You are a Senior Tech Recruiter and Hiring Match Engine.

Job Title: {job_title}
Company: {job_company}
Job Snippet / Description:
"{job_desc[:1200]}"

Candidate Skills: {skills_str}
Candidate Resume Snippet: "{resume_text[:500]}"

Analyze candidate suitability for this position.
Return ONLY valid JSON:
{{
    "match_score": 82,
    "fit_rating": "Strong Match / Moderate Fit / Needs Key Skills",
    "matched_skills": ["Skill1", "Skill2"],
    "missing_skills": ["Skill3", "Skill4"],
    "key_requirements": ["Req 1", "Req 2"],
    "verdict_summary": "2-sentence clear assessment of why the candidate fits or what gaps to bridge.",
    "application_tip": "Specific advice for tailoring resume bullet points for this position."
}}
"""
    try:
        raw = execute_groq_with_rotation(prompt, temperature=0.2, is_json=True, feature="career")
        return parse_insights_json(raw)
    except Exception as e:
        logger.error(f"Failed to evaluate job resume fit: {e}")
        return {
            "match_score": 75,
            "fit_rating": "Moderate Fit",
            "matched_skills": candidate_skills[:3] if candidate_skills else ["General CS"],
            "missing_skills": ["Role-specific domain tools"],
            "key_requirements": ["Core technical proficiency", "Communication"],
            "verdict_summary": f"Your technical foundation aligns well with the key needs for {job_title} at {job_company}.",
            "application_tip": "Emphasize past projects that utilize relevant frameworks."
        }


