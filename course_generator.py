"""Course generation logic — orchestrates Ollama AI (Local) + Groq AI (Cloud) + Invidious/Tavily Resource Searches."""

import json
import logging
import re
import requests
import httpx
from resource_search import search_youtube_tavily, search_documentation_tavily, search_web_duckduckgo
from tavily_helper import groq_client

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "deepseek-r1:1.5b"

SYSTEM_MSG = "You are an expert course creator. Return ONLY valid JSON. No preamble. No markdown. No conversational text. Do not explain your response."
BACKSLASH = chr(92)

# ─── Phase 1: Generate compact weekly outline ─────────────────
OUTLINE_PROMPT_TEMPLATE = """
Create a structured, industry-grade course outline for:
**Goal / Subject**: {goal}
**Difficulty Level**: {difficulty}
**Target Duration Strategy**: {duration_weeks_text}
**Learning Style**: {learning_style}

Return JSON:
{{
  "title": "Mastering {goal}: Complete Curriculum",
  "tagline": "Actionable, project-driven course roadmap tailored for {difficulty} level.",
  "difficulty": "{difficulty}",
  "duration_weeks": {suggested_weeks},
  "estimated_hours": {suggested_hours},
  "description": "Comprehensive course overview explaining what students will achieve.",
  "prerequisites": ["Prerequisite 1", "Prerequisite 2"],
  "learning_outcomes": [
    "Build enterprise-ready projects using modern tooling",
    "Master core theoretical & practical fundamentals",
    "Pass technical interview assessments for target roles"
  ],
  "capstone_project": "Description of the multi-week real-world final project.",
  "weeks": [
    {{
      "week": 1,
      "title": "Foundations & Environment Setup",
      "concepts": ["Core Syntax", "Development Tools"],
      "focus": "theory"
    }}
  ]
}}

RULES:
1. {weeks_rule}
2. Calculate "estimated_hours" accurately based on duration_weeks (approx. 10-15 study hours per week).
3. "prerequisites" MUST be a list of strings.
4. Output VALID JSON ONLY.
"""

# ─── Phase 2: Generate daily breakdown for a single week ──────
WEEK_DETAILS_PROMPT_TEMPLATE = """
Generate a comprehensive 7-day schedule breakdown for Week {week_number}: "{week_title}" of a course on "{goal}" ({difficulty} level).
Concepts to cover: {concepts}

PROGRESSION & DAY ALLOCATION RULES:
1. Generate EXACTLY 7 DAYS (Day 1 to Day 7).
2. Each day MUST have a distinct, unique topic or set of topics tailored specifically to {goal}.
3. If Week 1 Day 1: MUST cover Foundational Principles, Domain Architecture, and Environment Setup for {goal}.
4. If Week 1 Day 2: MUST cover Core Syntax, Primary Building Blocks, and Basic Operations for {goal}.
5. Days 3-5: Progressive core domain concepts, hands-on labs, and practical implementations for {goal} (different topics each day).
6. Day 6 (PRACTICE / REVISION): MUST be dedicated to Practice, Code Revision, Debugging & Refactoring Lab across Days 1-5 concepts.
7. Day 7 (CAPSTONE PROJECT): MUST be dedicated to building a Real-World Capstone Project Milestone applying the week's concepts.

Return JSON:
{{
  "days": [
    {{ "day": 1, "title": "Day 1: Introduction, Architecture & Environment Setup for {goal}", "task_type": "theory", "duration_minutes": 60, "concepts": ["{goal} Overview", "Environment Setup"] }},
    {{ "day": 2, "title": "Day 2: Core Syntax & Fundamental Building Blocks", "task_type": "practice", "duration_minutes": 90, "concepts": ["Core Building Blocks", "Basic Operations"] }},
    {{ "day": 3, "title": "Day 3: Control Flow & Applied Logic", "task_type": "practice", "duration_minutes": 60, "concepts": ["Domain Logic", "Control Structures"] }},
    {{ "day": 4, "title": "Day 4: Iteration & Algorithmic Patterns", "task_type": "practice", "duration_minutes": 90, "concepts": ["Iterative Operations", "Data Processing"] }},
    {{ "day": 5, "title": "Day 5: Functional & Modular Architecture", "task_type": "theory", "duration_minutes": 60, "concepts": ["Modular Design", "Best Practices"] }},
    {{ "day": 6, "title": "Day 6: Practice, Revision & Debugging Lab", "task_type": "practice", "duration_minutes": 120, "concepts": ["Weekly Revision", "Refactoring", "Edge Cases"] }},
    {{ "day": 7, "title": "Day 7: Real-World Capstone Project Milestone", "task_type": "project", "duration_minutes": 180, "concepts": ["Capstone Project Build"] }}
  ]
}}

RULES:
1. You MUST return ALL 7 DAYS.
2. Day 6 MUST be practice/revision. Day 7 MUST be project.
3. JSON ONLY.
"""


# ─── Phase 3: Generate details for a single day ──────────────
DAY_DETAILS_PROMPT_TEMPLATE = """
Generate learning content for Day {day_number}: "{day_title}".

**Course Goal**: {goal}
**Difficulty Level**: {difficulty}
**Task Type**: {task_type} ({duration_minutes} min)

TOPIC DENSITY & DAY-SPECIFIC INTENT RULES:
1. DYNAMIC TOPICS PER DAY: The number of topics can vary depending on concept depth and difficulty level ({difficulty}):
   - Beginner: 2-3 focused micro-topics.
   - Intermediate: 3-4 focused micro-topics.
   - Advanced: 4-5 focused micro-topics.
2. DAY INTENTIONS:
   - If Day 1: Focus topics on foundational principles, system setup, tools, and execution model for {goal}.
   - If Day 2: Focus topics on core building blocks, syntax, data structures, and primary operations for {goal}.
   - If Day 6 (PRACTICE / REVISION): Focus ALL topics on revision, hands-on practice challenges, edge-case debugging, and code refactoring for {goal}.
   - If Day 7 (CAPSTONE PROJECT): Focus ALL topics on building the end-to-end real-world capstone project step-by-step for {goal}.
3. For EACH topic, provide a distinct title, concise 1-2 sentence summary, code snippet, targeted YouTube search query, AND official documentation search query.

Return JSON:
{{
  "title": "{day_title}",
  "description": "Overview of today's learning objectives and key takeaways.",
  "topics": [
    {{
      "topic_name": "1. Distinct Topic Title",
      "summary": "Detailed 1-2 sentence explanation of this topic.",
      "code_snippet": "# Code snippet demonstration for this topic",
      "youtube_query": "{goal} topic tutorial short",
      "doc_query": "{goal} topic official reference guide"
    }}
  ],
  "hands_on_lab": {{
    "title": "Day {day_number} Hands-on Challenge",
    "instructions": "Step-by-step lab instructions.",
    "challenge": "Specific coding challenge.",
    "hint": "Useful implementation hint."
  }},
  "quiz": {{
    "question": "Question testing today's concepts?",
    "options": ["A) Option A", "B) Option B", "C) Option C", "D) Option D"],
    "hint": "Implementation hint.",
    "answer": "B) Option B",
    "explanation": "Detailed explanation."
  }}
}}

RULES:
1. Output VALID JSON ONLY.
"""



# ─── Ollama Utility Functions ──────────────────────────────────
def check_ollama_health() -> dict:
    """Check if local Ollama server is running."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=4.0)
        r.raise_for_status()
        return {"status": "connected", "url": OLLAMA_BASE_URL}
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return {"status": "disconnected", "url": OLLAMA_BASE_URL, "error": str(e)}


def list_models() -> list[dict]:
    """List locally available Ollama models."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models", []):
            models.append({
                "name": m.get("name", ""),
                "size": m.get("size", 0),
                "modified_at": m.get("modified_at", ""),
            })
        return models
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        return []


def call_ollama(model: str, prompt: str) -> str:
    """Call Ollama for text generation with timeout and system prompt."""
    model_name = model or DEFAULT_OLLAMA_MODEL
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 16384,
        },
    }
    logger.info(f"Calling Ollama model {model_name}...")
    r = requests.post(url, json=payload, timeout=120.0)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def call_ollama_chat(model: str, messages: list[dict]) -> str:
    """Call local Ollama Chat API synchronously for chatbot assistant or Groq fallback."""
    if not model or model == "groq" or model == "Groq AI (Cloud Llama 3.3)" or "groq" in model.lower():
        candidate_models = ["groq/compound", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "qwen/qwen3.6-27b"]
        for m in candidate_models:
            try:
                response = groq_client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq chat model {m} failed: {e}")
                continue

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }
    logger.info(f"Querying local Ollama Chat with model {model}...")
    response = requests.post(url, json=payload, timeout=60.0)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


from api_key_manager import execute_groq_with_rotation

# ─── Groq API Wrapper ─────────────────────────────────────────
def call_groq(prompt: str) -> str:
    """Query Groq API with multi-key rotation and feature-targeted key resolution."""
    return execute_groq_with_rotation(prompt, temperature=0.2, is_json=True, feature="coursegen")



def dispatch_llm_call(model_name: str, prompt: str) -> str:
    """Dispatch LLM request to Ollama if specified, else Groq cloud."""
    if model_name and (model_name.startswith("ollama:") or model_name not in ["groq", "Groq AI (Cloud Llama 3.3)", "llama-3.3-70b-versatile"]):
        clean_model = model_name.replace("ollama:", "")
        try:
            return call_ollama(clean_model, prompt)
        except Exception as e:
            logger.warning(f"Ollama call for '{clean_model}' failed ({e}), falling back to Groq AI Cloud...")
            return call_groq(prompt)
    else:
        return call_groq(prompt)


# ─── Robust JSON Parsing & Repair Engine (from ai-course-gen1) ───
def _unwrap_array(data):
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return data[0]
    return data

def _is_backslash(ch):
    return ch == BACKSLASH

def _find_last_complete_string_pos(text):
    last_closed_quote = -1
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if _is_backslash(ch):
            escape_next = True
            continue
        if ch == '"':
            if in_string:
                last_closed_quote = i
            in_string = not in_string
    return last_closed_quote

def _find_last_comma_outside_string(text):
    last_comma = -1
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if _is_backslash(ch):
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string and ch == ',':
            last_comma = i
    return last_comma

def _close_brackets(text):
    ob = text.count('{') - text.count('}')
    obr = text.count('[') - text.count(']')
    return text + ']' * max(0, obr) + '}' * max(0, ob)

def _try_parse(text):
    try:
        result = json.loads(text)
        return _unwrap_array(result)
    except (json.JSONDecodeError, ValueError):
        return None

def _try_repair_truncated_json(text):
    start = -1
    for i, ch in enumerate(text):
        if ch in ('{', '['):
            start = i
            break
    if start < 0:
        return None
    text = text[start:]

    # Attempt 0: Close open string
    quote_count = 0
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if _is_backslash(ch):
            escape_next = True
            continue
        if ch == '"':
            quote_count += 1
    
    if quote_count % 2 != 0:
        candidate = text + '"'
        result = _try_parse(_close_brackets(candidate))
        if result:
             return result

    # Attempt 1: Trim to last closed quote
    last_quote = _find_last_complete_string_pos(text)
    if last_quote > 0:
        candidate = text[:last_quote + 1]
        candidate = re.sub(r',\s*"[^"]*"\s*:\s*$', '', candidate)
        candidate = re.sub(r',\s*$', '', candidate)
        result = _try_parse(_close_brackets(candidate))
        if result:
            return result

    # Attempt 2: Trim to last comma
    last_comma = _find_last_comma_outside_string(text)
    if last_comma > 0:
        candidate = text[:last_comma]
        candidate = re.sub(r',\s*$', '', candidate)
        result = _try_parse(_close_brackets(candidate))
        if result:
            return result

    # Attempt 3: Trim to last }
    last_brace = text.rfind('}')
    if last_brace > 0:
        candidate = text[:last_brace + 1]
        candidate = re.sub(r',\s*$', '', candidate)
        result = _try_parse(_close_brackets(candidate))
        if result:
            return result

    return None

def parse_json_response(text):
    """Parse JSON from AI response handling reasoning tags, markdown fences, and truncation."""
    text = text.strip()

    # Strip <think> reasoning blocks (e.g., DeepSeek models)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if '<think>' in text:
        think_pos = text.find('<think>')
        json_start = text.find('{', think_pos)
        if json_start >= 0:
            text = text[:think_pos] + text[json_start:]
        else:
            text = text[:think_pos]
        text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    json_start = -1
    for i, ch in enumerate(text):
        if ch in ('{', '['):
            json_start = i
            break

    if json_start > 0:
        text = text[json_start:]
    elif json_start < 0:
        raise ValueError(f"No JSON found in AI response: {text[:200]}...")

    result = _try_parse(text)
    if result and isinstance(result, dict):
        return result

    repaired = _try_repair_truncated_json(text)
    if repaired and isinstance(repaired, dict):
        return repaired

    raise ValueError(f"Could not parse JSON from AI response: {text[:200]}...")


# ─── Subject Complexity & Timeline Auto-Detection Heuristic ────
def estimate_timeline_by_subject(goal: str, difficulty: str = "Intermediate") -> dict:
    """Determine recommended week count and estimated study hours based on subject complexity and difficulty."""
    goal_lower = (goal or "").strip().lower()
    diff_lower = (difficulty or "").strip().lower()

    # Domain keyword rules
    foundational_keywords = ["git", "github", "shell", "bash", "linux basics", "html", "css", "markdown", "terminal", "npm", "json", "regex"]
    core_keywords = ["python", "javascript", "react", "sql", "express", "django", "fastapi", "node", "java", "c++", "cpp", "c#", "typescript", "golang", "flutter"]
    complex_keywords = ["full stack", "fullstack", "machine learning", "deep learning", "ai", "data science", "devops", "kubernetes", "cloud", "aws", "system design", "data structures", "algorithms", "cybersecurity", "blockchain"]

    is_foundational = any(k in goal_lower for k in foundational_keywords)
    is_core = any(k in goal_lower for k in core_keywords)
    is_complex = any(k in goal_lower for k in complex_keywords)

    if is_foundational:
        base_weeks = 2 if diff_lower == "beginner" else 3
    elif is_complex:
        base_weeks = 8 if diff_lower == "beginner" else (12 if diff_lower == "advanced" else 10)
    elif is_core:
        base_weeks = 4 if diff_lower == "beginner" else (6 if diff_lower == "advanced" else 5)
    else:
        if diff_lower == "beginner":
            base_weeks = 4
        elif diff_lower == "advanced":
            base_weeks = 8
        else:
            base_weeks = 6

    hours_per_week = 12
    return {
        "weeks": base_weeks,
        "estimated_hours": base_weeks * hours_per_week,
        "reasoning": f"Subject '{goal}' evaluated as {'foundational tool' if is_foundational else ('complex domain' if is_complex else 'core skill')} for {difficulty} level."
    }


# ─── Main Course Generator Interface Functions ─────────────────
def generate_course_outline(goal: str, difficulty: str = "Intermediate", duration_weeks: str | int = "auto", learning_style: str = "Project-Based", model_name: str = None) -> dict:
    """Generate weekly course outline with dynamic subject/target timeline evaluation."""
    is_auto = str(duration_weeks).strip().lower() in ["auto", "0", "none", ""]
    auto_info = estimate_timeline_by_subject(goal, difficulty)

    if is_auto:
        dur_text = f"AUTO-DETECT optimal timeline based on subject/domain complexity of '{goal}' ({difficulty} level)"
        sug_weeks = auto_info["weeks"]
        sug_hours = auto_info["estimated_hours"]
        weeks_rule = f"Automatically calculate the optimal number of weeks ('duration_weeks') between 2 and 16 based on the depth of '{goal}' for {difficulty} level. Suggested baseline: {sug_weeks} weeks."
    else:
        try:
            target_num_weeks = int(duration_weeks)
        except (ValueError, TypeError):
            target_num_weeks = auto_info["weeks"]
        dur_text = f"{target_num_weeks} Weeks"
        sug_weeks = target_num_weeks
        sug_hours = target_num_weeks * 12
        weeks_rule = f"Generate exactly {target_num_weeks} weeks in the 'weeks' array."

    try:
        prompt = OUTLINE_PROMPT_TEMPLATE.format(
            goal=goal,
            difficulty=difficulty,
            duration_weeks_text=dur_text,
            suggested_weeks=sug_weeks,
            suggested_hours=sug_hours,
            learning_style=learning_style,
            weeks_rule=weeks_rule
        )
        raw = dispatch_llm_call(model_name, prompt)
        data = parse_json_response(raw)

        if "prerequisites" not in data or not isinstance(data["prerequisites"], list):
            data["prerequisites"] = ["Foundational Technical Knowledge", "Basic Problem Solving"]
        if "weeks" not in data or not isinstance(data["weeks"], list):
            data["weeks"] = []

        # Ensure week numbering is correct
        for idx, w in enumerate(data["weeks"]):
            w["week"] = idx + 1

        actual_weeks = len(data["weeks"]) or sug_weeks
        data["difficulty"] = difficulty
        data["duration_weeks"] = actual_weeks
        data["estimated_hours"] = data.get("estimated_hours") or (actual_weeks * 12)
        data["subject_target"] = goal
        return data
    except Exception as e:
        logger.error(f"Course outline generation failed: {e}")
        fallback_weeks = sug_weeks
        return {
            "title": f"Mastering {goal}: Complete Curriculum",
            "tagline": f"Actionable, project-driven course roadmap tailored for {difficulty} level.",
            "difficulty": difficulty,
            "estimated_hours": fallback_weeks * 12,
            "description": f"A structured curriculum designed to build end-to-end expertise in {goal}.",
            "prerequisites": ["Computer Science Fundamentals", "Basic Logic & Problem Solving"],
            "learning_outcomes": [
                f"Master core tools and techniques for {goal}",
                "Build scalable real-world portfolio projects",
                "Prepare for industry technical interviews"
            ],
            "capstone_project": f"Build a production-ready application demonstrating complete mastery of {goal}.",
            "weeks": [
                {
                    "week": i + 1,
                    "title": f"Phase {i+1}: Core & Advanced Modules for {goal}",
                    "concepts": [f"Core Topic {i+1}.1", f"Practical Tooling {i+1}.2"],
                    "focus": "practice" if i % 2 == 1 else "theory"
                } for i in range(fallback_weeks)
            ],
            "duration_weeks": fallback_weeks,
            "subject_target": goal
        }


def generate_week_details(goal: str, week_number: int, week_title: str, concepts: list[str], difficulty: str = "Intermediate", learning_style: str = "Project-Based", model_name: str = None, **kwargs) -> dict:
    """Generate 7-day daily breakdown for a specific week."""
    try:
        concepts_str = ", ".join(concepts) if concepts else week_title
        prompt = WEEK_DETAILS_PROMPT_TEMPLATE.format(
            goal=goal,
            week_number=week_number,
            week_title=week_title,
            concepts=concepts_str,
            difficulty=difficulty
        )
        raw = dispatch_llm_call(model_name, prompt)
        data = parse_json_response(raw)

        days = data.get("days", [])
        if not isinstance(days, list):
            days = []

        existing_nums = {d.get("day") for d in days if isinstance(d, dict) and d.get("day")}

        # Ensure all 7 days (Day 1 through Day 7) are present
        for day_num in range(1, 8):
            if day_num not in existing_nums:
                if day_num == 6:
                    days.append({
                        "day": 6,
                        "title": f"Day 6: Practice, Revision & Debugging Lab for {week_title}",
                        "task_type": "practice",
                        "duration_minutes": 120,
                        "concepts": [f"{week_title} Revision", "Code Refactoring", "Edge Case Debugging"],
                        "is_generated": True
                    })
                elif day_num == 7:
                    days.append({
                        "day": 7,
                        "title": f"Day 7: Real-World Capstone Project Milestone for {week_title}",
                        "task_type": "project",
                        "duration_minutes": 180,
                        "concepts": [f"{week_title} Capstone Project Build"],
                        "is_generated": True
                    })
                else:
                    days.append({
                        "day": day_num,
                        "title": f"Day {day_num}: {week_title} Module {day_num}",
                        "task_type": "practice" if day_num % 2 == 0 else "theory",
                        "duration_minutes": 60,
                        "concepts": [f"{week_title} Topic {day_num}"],
                        "is_generated": True
                    })

        for d in days:
            if "concepts" not in d or not d["concepts"]:
                d["concepts"] = [f"{week_title} Core Concept"]
            d["is_generated"] = True

        days = sorted(days, key=lambda x: x.get("day", 1))
        data["days"] = days
        return data
    except Exception as e:
        logger.error(f"Week details generation failed: {e}")
        c_list = concepts if concepts else [f"{week_title} Overview", f"{week_title} Applied Concepts"]
        return {
            "days": [
                {
                    "day": 1,
                    "title": f"Day 1: Introduction, Architecture & Environment Setup for {goal}",
                    "task_type": "theory",
                    "duration_minutes": 60,
                    "concepts": [c_list[0] if c_list else f"{goal} Overview", f"{goal} Setup"],
                    "is_generated": True
                },
                {
                    "day": 2,
                    "title": f"Day 2: Core Fundamentals & Building Blocks of {week_title}",
                    "task_type": "practice",
                    "duration_minutes": 90,
                    "concepts": [c_list[1] if len(c_list) > 1 else week_title, "Core Syntax"],
                    "is_generated": True
                },
                {
                    "day": 3,
                    "title": f"Day 3: Control Flow & Logic in {week_title}",
                    "task_type": "practice",
                    "duration_minutes": 60,
                    "concepts": [f"{week_title} Logic & Conditionals"],
                    "is_generated": True
                },
                {
                    "day": 4,
                    "title": f"Day 4: Iteration & Algorithm Structures in {week_title}",
                    "task_type": "practice",
                    "duration_minutes": 90,
                    "concepts": [f"{week_title} Loops & Algorithms"],
                    "is_generated": True
                },
                {
                    "day": 5,
                    "title": f"Day 5: Modular Design & System Patterns for {week_title}",
                    "task_type": "theory",
                    "duration_minutes": 60,
                    "concepts": [f"{week_title} Design Patterns"],
                    "is_generated": True
                },
                {
                    "day": 6,
                    "title": f"Day 6: Practice, Revision & Debugging Lab for {week_title}",
                    "task_type": "practice",
                    "duration_minutes": 120,
                    "concepts": [f"{week_title} Revision & Refactoring"],
                    "is_generated": True
                },
                {
                    "day": 7,
                    "title": f"Day 7: Real-World Capstone Project Milestone for {week_title}",
                    "task_type": "project",
                    "duration_minutes": 180,
                    "concepts": [f"{week_title} Capstone Project Build"],
                    "is_generated": True
                }
            ]
        }



def generate_day_details(goal: str, day_title: str, day_number: int, task_type: str, duration_minutes: int, difficulty: str = "Intermediate", learning_style: str = "Project-Based", model_name: str = None, **kwargs) -> dict:
    """Generate detailed lesson micro-topics, code snippets, labs, quiz & targeted video/doc resources."""
    try:
        prompt = DAY_DETAILS_PROMPT_TEMPLATE.format(
            goal=goal,
            day_title=day_title,
            day_number=day_number,
            task_type=task_type,
            duration_minutes=duration_minutes,
            difficulty=difficulty
        )
        raw = dispatch_llm_call(model_name, prompt)
        data = parse_json_response(raw)
    except Exception as e:
        logger.error(f"Day details generation failed: {e}")
        data = {
            "title": day_title,
            "description": f"Comprehensive educational breakdown covering {day_title} in {goal}.",
            "topics": []
        }

    # Enforce strict progression for Day 1 and Day 2 if Goal relates to Python or Programming
    is_python_prog = any(k in goal.lower() for k in ["python", "code", "programming", "developer", "software"])
    if day_number == 1 and is_python_prog:
        data["topics"] = [
            {
                "topic_name": "1. Compilers vs. Interpreters",
                "summary": "Explains how source code is translated to machine instructions via compilation vs line-by-line interpretation.",
                "code_snippet": "# Python bytecode disassembly example\nimport dis\n\ndef demo():\n    x = 10\n    y = 20\n    return x + y\n\ndis.dis(demo)",
                "youtube_query": "Compilers vs Interpreters programming tutorial",
                "doc_query": "CPython bytecode virtual machine execution model"
            },
            {
                "topic_name": "2. CPython & Execution Virtual Machine",
                "summary": "How Python source code (.py) is compiled into bytecode (.pyc) and run by the Python Virtual Machine (PVM).",
                "code_snippet": "import sys\nprint('Python Version:', sys.version)\nprint('Executable Path:', sys.executable)",
                "youtube_query": "How CPython works bytecode PVM tutorial",
                "doc_query": "sys module python runtime reference"
            }
        ]
    elif day_number == 2 and is_python_prog:
        data["topics"] = [
            {
                "topic_name": "1. Variables & Primitive Data Types",
                "summary": "Declaring variables, dynamic typing, and memory references for integers, floats, booleans, and strings.",
                "code_snippet": "age = 25              # int\nheight = 5.9          # float\nname = 'Alice'        # str\nis_student = True     # bool\n\nprint(type(age), type(height), type(name), type(is_student))",
                "youtube_query": "Python variables and primitive data types tutorial",
                "doc_query": "Python Built-in Types official documentation"
            },
            {
                "topic_name": "2. Printing & Input Formatting",
                "summary": "Mastering console output formatting with f-strings, str.format(), and reading user input with input().",
                "code_snippet": "name = input('Enter your name: ')\nscore = 98.5\n# Modern f-string formatting\nprint(f'Hello {name}, your test score is {score:.1f}%!')",
                "youtube_query": "Python print formatting f-strings input tutorial",
                "doc_query": "Python formatted string literals f-strings guide"
            }
        ]
    elif day_number == 6:
        data["title"] = day_title or f"Day 6: Practice, Revision & Debugging Lab for {goal}"
        if "topics" not in data or not data["topics"]:
            data["topics"] = [
                {
                    "topic_name": f"1. Comprehensive Concept Revision ({day_title})",
                    "summary": "Hands-on review and memory checkpoint covering all topics learned from Days 1 to 5.",
                    "code_snippet": f"# Revision Lab: Inspecting core logic for {goal}\ndef revision_checkpoint():\n    return 'Days 1-5 concepts verified'",
                    "youtube_query": f"{goal} practice exercises revision tutorial",
                    "doc_query": f"{goal} best practices reference guide"
                },
                {
                    "topic_name": "2. Speed Coding & Edge-Case Debugging",
                    "summary": "Solving real-world edge cases, handling exceptions, and optimizing runtime execution.",
                    "code_snippet": "try:\n    # Edge-case validation\n    value = int(input('Enter score: '))\nexcept ValueError:\n    value = 0\nprint('Processed score:', value)",
                    "youtube_query": f"{goal} debugging edge cases practice tutorial",
                    "doc_query": f"{goal} error handling reference"
                },
                {
                    "topic_name": "3. Code Refactoring & Optimization Lab",
                    "summary": "Refactoring raw scripts into modular, production-ready functions and PEP8 standards.",
                    "code_snippet": "# Refactored clean helper\ndef clean_inputs(raw_data):\n    return [item.strip().lower() for item in raw_data if item]",
                    "youtube_query": f"{goal} clean code refactoring tutorial",
                    "doc_query": f"{goal} code style optimization"
                }
            ]
        data["hands_on_lab"] = {
            "title": f"Day 6 Practice & Revision Lab Challenge: {day_title}",
            "instructions": "Refactor a buggy script, fix all edge-case exceptions, and write clean unit tests.",
            "challenge": "Refactor a monolithic script into 3 modular functions with full error handling.",
            "hint": "Focus on input validation, PEP8 formatting, and exception handling."
        }
    elif day_number == 7:
        data["title"] = day_title or f"Day 7: Real-World Capstone Project Milestone for {goal}"
        if "topics" not in data or not data["topics"]:
            data["topics"] = [
                {
                    "topic_name": f"1. Capstone Project Architecture ({day_title})",
                    "summary": f"Designing system architecture and data structures for the {goal} capstone project.",
                    "code_snippet": f"# Capstone Application Skeleton\nclass CapstoneApplication:\n    def __init__(self, app_name):\n        self.app_name = app_name\n        self.modules = []",
                    "youtube_query": f"{goal} full project step by step tutorial",
                    "doc_query": f"{goal} project architecture design patterns"
                },
                {
                    "topic_name": "2. Core Feature Implementation & Module Integration",
                    "summary": "Integrating modules built throughout the week into a unified functional capstone application.",
                    "code_snippet": "app = CapstoneApplication('Enterprise Module')\nprint(f'Building {app.app_name}...')",
                    "youtube_query": f"{goal} build real world app tutorial",
                    "doc_query": f"{goal} application integration guide"
                },
                {
                    "topic_name": "3. Testing, Verification & Project Deployment",
                    "summary": "Running automated unit tests, verifying output correctness, and packaging final deliverables.",
                    "code_snippet": "def test_app():\n    assert app.app_name == 'Enterprise Module'\n    print('Capstone project passed all verification tests!')\ntest_app()",
                    "youtube_query": f"{goal} testing and deployment tutorial",
                    "doc_query": f"{goal} deployment verification guide"
                }
            ]
        data["hands_on_lab"] = {
            "title": f"Day 7 Real-World Capstone Project Milestone: {day_title}",
            "instructions": "Build and deploy a complete production-grade application integrating all concepts learned this week.",
            "challenge": "Create a fully functional application that processes data, handles exceptions, and generates outputs.",
            "hint": "Incorporate modular class architecture, error handling, and clean documentation."
        }

    # Ensure topic fallback if AI returned empty topics list
    if "topics" not in data or not data["topics"]:
        data["topics"] = [
            {
                "topic_name": f"1. Core Principles of {day_title}",
                "summary": f"Essential theoretical concepts underpinning {day_title}.",
                "code_snippet": f"# Topic 1: {day_title}\ndef topic_one():\n    print('Topic 1 initialized')",
                "youtube_query": f"{goal} {day_title} core principles tutorial",
                "doc_query": f"{goal} {day_title} official guide"
            },
            {
                "topic_name": f"2. Practical Implementation of {day_title}",
                "summary": f"Hands-on tutorial and code patterns for {day_title}.",
                "code_snippet": f"# Topic 2: {day_title}\ndef topic_two():\n    print('Topic 2 executed')",
                "youtube_query": f"{goal} {day_title} practical hands on tutorial",
                "doc_query": f"{goal} {day_title} reference documentation"
            }
        ]


    # Perform dedicated YouTube AND Documentation Search for EACH micro-topic
    for topic in data.get("topics", []):
        t_name = topic.get("topic_name", day_title)
        sq_yt = topic.get("youtube_query") or f"{goal} {t_name} tutorial"
        sq_doc = topic.get("doc_query") or f"{goal} {t_name} documentation"

        # 1. Fetch dedicated YouTube video for this specific topic
        try:
            yt_res = search_youtube_tavily(sq_yt, max_results=1)
            if yt_res:
                topic["video"] = yt_res[0]
            else:
                topic["video"] = {
                    "title": f"Watch {t_name} Tutorial on YouTube",
                    "url": f"https://www.youtube.com/results?search_query={sq_yt.replace(' ', '+')}",
                    "embed_url": f"https://www.youtube.com/results?search_query={sq_yt.replace(' ', '+')}",
                    "source": "youtube",
                    "thumbnail": None
                }
        except Exception as search_err:
            logger.error(f"Topic video search failed for {sq_yt}: {search_err}")
            topic["video"] = {
                "title": f"Watch {t_name} Tutorial on YouTube",
                "url": f"https://www.youtube.com/results?search_query={sq_yt.replace(' ', '+')}",
                "embed_url": f"https://www.youtube.com/results?search_query={sq_yt.replace(' ', '+')}",
                "source": "youtube",
                "thumbnail": None
            }

        # 2. Fetch dedicated Official Documentation link for this specific topic
        try:
            doc_res = search_documentation_tavily(sq_doc, max_results=1)
            if doc_res:
                topic["doc"] = doc_res[0]
            else:
                topic["doc"] = {
                    "title": f"{t_name} Reference Guide",
                    "url": f"https://duckduckgo.com/?q={sq_doc.replace(' ', '+')}",
                    "source": "documentation"
                }
        except Exception as doc_err:
            logger.error(f"Topic doc search failed for {sq_doc}: {doc_err}")
            topic["doc"] = {
                "title": f"{t_name} Reference Guide",
                "url": f"https://duckduckgo.com/?q={sq_doc.replace(' ', '+')}",
                "source": "documentation"
            }

    # Fallback hands-on lab & quiz if missing
    if "hands_on_lab" not in data or not data["hands_on_lab"]:
        data["hands_on_lab"] = {
            "title": f"Hands-on Lab: {day_title}",
            "instructions": f"Implement a clean python script applying concepts from {day_title}.",
            "challenge": "Build a module that processes inputs and handles exceptions.",
            "hint": "Refer to code snippets provided in today's topics."
        }

    if "quiz" not in data or not data["quiz"]:
        data["quiz"] = {
            "question": f"What is the primary objective when mastering {day_title}?",
            "options": [
                "Option A: Ensuring clean, maintainable, and correct code execution",
                "Option B: Hardcoding static values without variable assignments",
                "Option C: Bypassing interpreter runtime checks",
                "Option D: Disabling environment isolation"
            ],
            "hint": "Think about standard software engineering principles.",
            "answer": "Option A: Ensuring clean, maintainable, and correct code execution",
            "explanation": "Mastery of fundamental execution principles guarantees code stability and scalability."
        }

    return data
