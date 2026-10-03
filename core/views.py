import os
import json
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .apps import CoreConfig

# Helper imports from root codebase
from resume_parser import analyze_resume_profile, suggest_new_role_ai
from skill_mapper import load_database, get_role_skills, save_database
from tavily_helper import (
    SUB_TO_PARENT_ROLE,
    get_career_why_and_what,
    get_job_market_insights,
    get_active_jobs_and_internships
)
from recommendation_cache import process_dual_recommendations, get_ai_job_recommendation
from course_generator import (
    check_ollama_health,
    list_models,
    generate_course_outline,
    generate_week_details,
    generate_day_details,
    call_ollama_chat
)

SPECIALIZED_TO_STANDARD_MAPPING = {
    'ACCESSIBILITY SPECIALIST': 'Software Engineer',
    'AGILE PROJECT MANAGER': 'Analyst',
    'BUSINESS SYSTEMS ANALYST': 'Analyst',
    'CLOUD ARCHITECT': 'Software Engineer',
    'COMPUTER GRAPHICS ANIMATOR': 'Web Developer',
    'DATA ANALYST': 'Analyst',
    'DATA MODELER': 'Data Scientist',
    'DATA SCIENTIST': 'Data Scientist',
    'DEVOPS MANAGER': 'Software Engineer',
    'FRAMEWORKS SPECIALIST': 'Software Engineer',
    'INFORMATION ARCHITECT': 'Analyst',
    'INTERACTION DESIGNER': 'Web Developer',
    'MOBILE APP DEVELOPER': 'Software Engineer',
    'PRODUCT MANAGER': 'Analyst',
    'SECURITY SPECIALIST': 'Software Engineer',
    'TECHNICAL ACCOUNT MANAGER': 'Analyst',
    'TECHNICAL LEAD': 'Software Engineer'
}


def dashboard_view(request):
    """Render the main fullstack dark glassmorphic HTML dashboard."""
    context = {
        'branch_options': CoreConfig.branch_options,
        'profile_model_active': CoreConfig.ml_model is not None,
        'skill_model_active': CoreConfig.skill_model is not None,
    }
    return render(request, 'index.html', context)


@csrf_exempt
def parse_resume_api(request):
    """Handle resume (PDF/DOCX) file uploads and return extracted skills."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed'}, status=405)

    uploaded_file = request.FILES.get('resume')
    if not uploaded_file:
        return JsonResponse({'status': 'error', 'message': 'No resume file uploaded'}, status=400)

    try:
        resume_profile = analyze_resume_profile(uploaded_file)
        skills_extracted = resume_profile.get("skills", [])
        resume_text = resume_profile.get("resume_text", "")
        return JsonResponse({
            'status': 'success',
            'skills': skills_extracted,
            'skills_str': ", ".join(skills_extracted),
            'resume_text': resume_text
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def analyze_career_api(request):
    """Run dual ML models & ATS skill mapping to generate career recommendations."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed'}, status=405)

    try:
        # Support both JSON body and form-data
        if request.content_type == 'application/json':
            data = json.loads(request.body.decode('utf-8'))
        else:
            data = request.POST

        branch = data.get('branch', 'CSE')
        cgpa = float(data.get('cgpa', 8.0))
        college_tier = int(data.get('college_tier', 2))
        coding_score = float(data.get('coding_score', 75))
        aptitude_score = float(data.get('aptitude_score', 75))
        communication_score = float(data.get('communication_score', 8))
        internships = int(data.get('internships', 1))
        projects = int(data.get('projects', 2))
        backlogs = int(data.get('backlogs', 0))
        dsa_skill = bool(data.get('dsa_skill', True))
        skills_input = data.get('skills_input', '')
        resume_text = data.get('resume_text', '')

        if not skills_input.strip():
            return JsonResponse({'status': 'error', 'message': 'Technical skills field is required.'}, status=400)

        combined_skills = [s.strip() for s in skills_input.split(",") if s.strip()]
        combined_skills_str = ", ".join(combined_skills)
        combined_skills_set = {s.lower() for s in combined_skills}

        skill_score = round(min(0.5 + (len(combined_skills) * 0.25), 3.5), 2)
        resume_score = round(min(max(30.0 + (cgpa * 4) + (len(combined_skills) * 2) + (internships * 5) + (projects * 3) - (backlogs * 5), 22.5), 130.0), 2)

        # 1. Profile ML Model prediction
        ml_probabilities = {}
        max_ml_confidence = 0.0
        if CoreConfig.ml_model and CoreConfig.ml_encoders:
            try:
                branch_enc = CoreConfig.ml_encoders['branch'].transform([branch])[0]
                input_df = pd.DataFrame([[
                    cgpa, branch_enc, college_tier, int(dsa_skill), coding_score,
                    communication_score, aptitude_score, internships, projects,
                    backlogs, resume_score, skill_score
                ]], columns=[
                    'cgpa', 'branch', 'college_tier', 'dsa_skill', 'coding_score',
                    'communication_score', 'aptitude_score', 'internships', 'projects',
                    'backlogs', 'resume_score', 'skill_score'
                ])
                probabilities = CoreConfig.ml_model.predict_proba(input_df)[0]
                role_classes = CoreConfig.ml_encoders['job_role'].classes_
                for rc, prob in zip(role_classes, probabilities):
                    ml_probabilities[rc] = float(prob * 100)
                if ml_probabilities:
                    max_ml_confidence = max(ml_probabilities.values())
            except Exception as ml_err:
                print(f"[WARN] Profile classifier failed: {ml_err}")

        # 2. Database skill matching
        db = load_database()
        db_match_scores = {}
        for role, data_val in db.items():
            req_skills = data_val.get("required_skills", [])
            req_set = {s.lower() for s in req_skills}
            matched_set = req_set & combined_skills_set
            score = (len(matched_set) / len(req_skills)) * 100 if req_skills else 0
            db_match_scores[role] = score

        max_db_match = max(db_match_scores.values()) if db_match_scores else 0.0

        # 3. AI Fallback Logic
        ai_fallback_triggered = False
        ai_suggested_role = None

        if max_ml_confidence < 35.0 and max_db_match < 30.0:
            ai_suggestion = suggest_new_role_ai(resume_text or combined_skills_str, combined_skills)
            ai_suggested_role = ai_suggestion.get("suggested_role", "Software Engineer").strip()
            ai_suggested_skills = ai_suggestion.get("required_skills", [])

            if ai_suggested_role not in db:
                db[ai_suggested_role] = {"required_skills": ai_suggested_skills}
                save_database(db)
                db = load_database()

            db_match_scores[ai_suggested_role] = (len({s.lower() for s in ai_suggested_skills} & combined_skills_set) / len(ai_suggested_skills)) * 100 if ai_suggested_skills else 0
            ai_fallback_triggered = True

        # 4. Specialized Skill Classifier (17 roles)
        skill_recs = []
        if CoreConfig.skill_model and CoreConfig.skill_tfidf and CoreConfig.skill_encoder:
            skills_vector = CoreConfig.skill_tfidf.transform([combined_skills_str])
            probabilities = CoreConfig.skill_model.predict_proba(skills_vector)[0]
            role_classes = CoreConfig.skill_encoder.classes_

            skill_probabilities = {}
            for rc, prob in zip(role_classes, probabilities):
                skill_probabilities[rc] = float(prob * 100)

            for role in role_classes:
                req_skills = get_role_skills(role)
                req_set = {s.lower() for s in req_skills}
                matched_set = req_set & combined_skills_set

                skill_match_score = (len(matched_set) / len(req_skills)) * 100 if req_skills else 0
                skills_ml_score = skill_probabilities.get(role, 0.0)

                mapped_std_role = SPECIALIZED_TO_STANDARD_MAPPING.get(role, "Software Engineer")
                profile_prob = ml_probabilities.get(mapped_std_role, 0.0) if ml_probabilities else 0.0

                combined_score = round((0.4 * skills_ml_score) + (0.3 * skill_match_score) + (0.3 * profile_prob), 2)
                parent_role = SUB_TO_PARENT_ROLE.get(role.upper(), None)

                skill_recs.append({
                    "role": role,
                    "parent_role": parent_role,
                    "score": combined_score,
                    "skills_ml_score": round(skills_ml_score, 2),
                    "skill_score": round(skill_match_score, 2),
                    "profile_score": round(profile_prob, 2),
                    "required_skills": req_skills,
                    "matched_skills": [s for s in req_skills if s.lower() in matched_set],
                    "missing_skills": [s for s in req_skills if s.lower() not in matched_set]
                })

            skill_recs = sorted(skill_recs, key=lambda x: x["score"], reverse=True)

        # Determine Top ML Role from prediction model
        top_ml_role = skill_recs[0]["role"] if skill_recs else "Software Engineer"

        # Execute Dual Recommendation Engine (ML Model vs Groq AI + Discrepancy Cache Learning)
        profile_dict = {
            "branch": branch,
            "cgpa": cgpa,
            "college_tier": college_tier,
            "coding_score": coding_score,
            "aptitude_score": aptitude_score,
            "communication_score": communication_score,
            "internships": internships,
            "projects": projects,
            "backlogs": backlogs,
            "dsa_skill": dsa_skill,
        }
        dual_insights = process_dual_recommendations(
            ml_top_role=top_ml_role,
            profile_data=profile_dict,
            skills=combined_skills
        )

        return JsonResponse({
            'status': 'success',
            'combined_skills': combined_skills,
            'skill_recs': skill_recs,
            'ml_probabilities': ml_probabilities,
            'db_match_scores': db_match_scores,
            'ai_fallback_triggered': ai_fallback_triggered,
            'ai_suggested_role': ai_suggested_role,
            'dual_insights': dual_insights
        })


    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


# Role-specific in-memory caching
ROLE_INSIGHTS_CACHE = {}
ROLE_WHY_WHAT_CACHE = {}

@csrf_exempt
def why_what_api(request):
    """Fetch live 'Why & What Insights' for a given role and skills with role-keyed caching."""
    role = request.GET.get('role') or request.POST.get('role')
    skills_raw = request.GET.get('skills') or request.POST.get('skills') or ''

    if not role:
        return JsonResponse({'status': 'error', 'message': 'Role parameter required'}, status=400)

    skills = [s.strip() for s in skills_raw.split(',') if s.strip()] if isinstance(skills_raw, str) else skills_raw
    cache_key = f"{role.upper()}_{'_'.join(sorted(skills[:5]))}"

    if cache_key in ROLE_WHY_WHAT_CACHE:
        return JsonResponse({'status': 'success', 'insights': ROLE_WHY_WHAT_CACHE[cache_key]})

    try:
        insights = get_career_why_and_what(role, skills)
        ROLE_WHY_WHAT_CACHE[cache_key] = insights
        return JsonResponse({'status': 'success', 'insights': insights})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def market_insights_api(request):
    """Fetch live job market insights via Tavily with strict role-keyed caching."""
    role = request.GET.get('role') or request.POST.get('role')
    if not role:
        return JsonResponse({'status': 'error', 'message': 'Role parameter required'}, status=400)

    role_key = role.upper().strip()
    if role_key in ROLE_INSIGHTS_CACHE:
        return JsonResponse({'status': 'success', 'insights': ROLE_INSIGHTS_CACHE[role_key]})

    try:
        insights = get_job_market_insights(role)
        ROLE_INSIGHTS_CACHE[role_key] = insights
        return JsonResponse({'status': 'success', 'insights': insights})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def live_jobs_api(request):
    """Fetch active jobs & internships via Tavily for the specified role."""
    role = request.GET.get('role') or request.POST.get('role')
    if not role:
        return JsonResponse({'status': 'error', 'message': 'Role parameter required'}, status=400)

    try:
        jobs_data = get_active_jobs_and_internships(role)
        return JsonResponse({'status': 'success', 'data': jobs_data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def course_outline_api(request):
    """Generate multi-week AI course outline via Groq or Ollama with difficulty, duration, and model selection."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
        except Exception:
            data = request.POST
    else:
        data = request.GET

    goal = data.get('learning_goal') or data.get('goal', 'Learn Software Development')
    difficulty = data.get('difficulty', 'Intermediate')
    duration_weeks = data.get('duration_weeks', 'auto')
    learning_style = data.get('learning_style', 'Project-Based')
    model_name = data.get('model') or data.get('model_name')

    try:
        outline = generate_course_outline(
            goal=goal,
            difficulty=difficulty,
            duration_weeks=duration_weeks,
            learning_style=learning_style,
            model_name=model_name
        )
        return JsonResponse({'status': 'success', 'outline': outline, **outline})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def week_details_api(request):
    """Generate daily tasks breakdown for a course week."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        goal = data.get('learning_goal') or data.get('goal')
        week_num = int(data.get('week_num') or data.get('week_number', 1))
        week_title = data.get('week_title', f'Week {week_num}')
        concepts = data.get('concepts', [])
        difficulty = data.get('difficulty', 'Intermediate')
        learning_style = data.get('learning_style', 'Project-Based')
        model_name = data.get('model') or data.get('model_name')

        w_data = generate_week_details(goal, week_num, week_title, concepts, difficulty=difficulty, learning_style=learning_style, model_name=model_name)
        return JsonResponse({'status': 'success', 'week_details': w_data, **w_data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def day_details_api(request):
    """Generate detailed lesson, code snippet, coding lab, quiz & resource links for a single day."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        goal = data.get('learning_goal') or data.get('goal')
        day_title = data.get('day_title')
        day_num = int(data.get('day_num') or data.get('day_number', 1))
        task_type = data.get('task_type', 'theory')
        duration = int(data.get('duration_minutes', 60))
        difficulty = data.get('difficulty', 'Intermediate')
        learning_style = data.get('learning_style', 'Project-Based')
        model_name = data.get('model') or data.get('model_name')

        d_data = generate_day_details(goal, day_title, day_num, task_type, duration, difficulty=difficulty, learning_style=learning_style, model_name=model_name)
        return JsonResponse({'status': 'success', 'day_details': d_data, **d_data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)



@csrf_exempt
def models_api(request):
    """Return list of locally installed Ollama models for ai-course-gen1 compatibility."""
    try:
        models = list_models()
        return JsonResponse({'status': 'success', 'models': models})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def health_api(request):
    """Health check endpoint for ai-course-gen1 compatibility."""
    try:
        health = check_ollama_health()
        return JsonResponse({'status': 'healthy', 'ollama': health})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def ollama_status_api(request):
    """Check health and retrieve models from local Ollama server."""
    try:
        health = check_ollama_health()
        models = list_models() if health.get('status') == 'connected' else []
        return JsonResponse({
            'status': 'success',
            'connected': health.get('status') == 'connected',
            'health': health,
            'models': models
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'connected': False, 'message': str(e)})


@csrf_exempt
def ollama_chat_api(request):
    """Chat interactive endpoint connected to local Ollama model or Groq cloud."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        model = data.get('model')
        title = data.get('title', 'Course')
        goal = data.get('goal', 'Learning')
        messages = data.get('messages', [])

        chat_payload = [
            {"role": "system", "content": "You are a professional educational tutor."},
            {"role": "system", "content": f"Course: '{title}' Goal: '{goal}'."}
        ] + messages[-8:]

        response_text = call_ollama_chat(model, chat_payload)
        return JsonResponse({'status': 'success', 'reply': response_text})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


