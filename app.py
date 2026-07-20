# -*- coding: utf-8 -*-
import streamlit as st
import os
import json
import pickle
import pandas as pd
import joblib
from resume_parser import extract_resume_text, analyze_resume_profile, suggest_new_role_ai
#from ats_engine import extract_resume_skills
from skill_mapper import load_database
from tavily_helper import SUB_TO_PARENT_ROLE
from course_generator import (
    check_ollama_health,
    list_models,
    generate_course_outline,
    generate_week_details,
    generate_day_details,
    call_ollama_chat
)


# Ensure environment variables are loaded
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"\'')

# Set page configuration with premium UI setup
st.set_page_config(
    page_title="AI Hybrid Job Recommender & ATS Gap Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load machine learning model assets
@st.cache_resource
def load_ml_assets():
    try:
        with open("profile_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("profile_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        return model, encoders
    except Exception as e:
        st.error(f"Error loading Profile ML model/encoders: {str(e)}")
        return None, None

@st.cache_resource
def load_skill_ml_assets():
    try:
        model = joblib.load("job_role_model.pkl")
        tfidf = joblib.load("tfidf.pkl")
        encoder = joblib.load("label_encoder.pkl")
        return model, tfidf, encoder
    except Exception as e:
        st.error(f"Error loading Skill ML assets (job_role_model.pkl, label_encoder.pkl, tfidf.pkl): {str(e)}")
        return None, None, None

ml_model, ml_encoders = load_ml_assets()
skill_model, skill_tfidf, skill_encoder = load_skill_ml_assets()

if ml_encoders:
    branch_options = list(ml_encoders['branch'].classes_)
    job_role_classes = list(ml_encoders['job_role'].classes_)
else:
    branch_options = ['CSE', 'Civil', 'ECE', 'EEE', 'IT', 'Mechanical']
    job_role_classes = ['Analyst', 'Data Scientist', 'Software Engineer', 'Web Developer']

# Premium stylesheet injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Background styles */
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, #030712 60%);
        color: #f3f4f6;
    }
    
    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7);
        margin-bottom: 24px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
    }
    
    /* Glowing main header */
    .main-title {
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 10px;
        letter-spacing: -0.05em;
    }
    
    .subtitle {
        color: #9ca3af;
        text-align: center;
        font-size: 1.15rem;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    /* Form inputs styling */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background-color: rgba(31, 41, 55, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f3f4f6 !important;
        border-radius: 10px !important;
        padding: 10px 15px !important;
    }
    
    /* Button styles */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px -2px rgba(99, 102, 241, 0.5) !important;
        width: 100%;
        letter-spacing: 0.02em;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px -2px rgba(99, 102, 241, 0.7) !important;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
    }
    
    /* Pill Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 5px;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
    }
    
    .badge-matched {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    
    .badge-matched:hover {
        background: rgba(16, 185, 129, 0.2);
    }
    
    .badge-missing {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    
    .badge-missing:hover {
        background: rgba(239, 68, 68, 0.2);
    }
    
    .badge-normal {
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }
    
    /* Circular ATS Score Gauge */
    .score-circle-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 20px 0;
    }
    
    .score-circle {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 140px;
        height: 140px;
        border-radius: 50%;
        font-size: 2.5rem;
        font-weight: 800;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .score-high {
        background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.03) 100%);
        border: 5px solid #10b981;
        color: #10b981;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.25);
    }
    
    .score-mid {
        background: radial-gradient(circle, rgba(245, 158, 11, 0.15) 0%, rgba(245, 158, 11, 0.03) 100%);
        border: 5px solid #f59e0b;
        color: #f59e0b;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.25);
    }
    
    .score-low {
        background: radial-gradient(circle, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.03) 100%);
        border: 5px solid #ef4444;
        color: #ef4444;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.25);
    }
    
    .score-label {
        font-weight: 700;
        font-size: 1rem;
        margin-top: 15px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Recommendation Bar Chart Styling */
    .rec-item {
        margin-bottom: 16px;
    }
    
    .rec-label-container {
        display: flex;
        justify-content: space-between;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }
    
    .rec-bar-bg {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        height: 12px;
        width: 100%;
        overflow: hidden;
    }
    
    .rec-bar-fill {
        background: linear-gradient(90deg, #6366f1 0%, #a5b4fc 100%);
        border-radius: 9999px;
        height: 100%;
        transition: width 1s ease-in-out;
    }
    
    .rec-bar-fill-top {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Main Application Layout
st.markdown('<div class="main-title">AI Hybrid Job Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predicts Placement Opportunities via Profile (profile_model.pkl) & Career Paths via Technical Skills (job_role_model.pkl, label_encoder.pkl, tfidf.pkl)</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Fallback Logic Rules")
    st.info("The **AI Career Advisor** triggers only if the local ML model cannot predict a path confidently (<35%) AND the database contains no high-matching roles (<30%).")
    
    st.markdown("### 📊 Engine Status")
    if ml_model:
        st.success("✅ Placement Predictor (profile_model.pkl): Active")
    else:
        st.error("❌ Placement Predictor (profile_model.pkl): Offline")
        
    if skill_model:
        st.success("✅ Career Path Recommender (job_role_model.pkl): Active")
    else:
        st.error("❌ Career Path Recommender (job_role_model.pkl): Offline")
    st.success("✅ AI Fallback Suggestion: Enabled")
    
    st.markdown("---")
    if st.button("🔄 Clear App Cache"):
        st.cache_data.clear()
        st.success("Cache cleared successfully!")
        st.rerun()
        
    st.markdown("🔒 *Processing is private and temporary.*")

# Initialize session state for resume extraction
if "last_uploaded_file_key" not in st.session_state:
    st.session_state["last_uploaded_file_key"] = None
if "extracted_skills_str" not in st.session_state:
    st.session_state["extracted_skills_str"] = ""
if "resume_text" not in st.session_state:
    st.session_state["resume_text"] = ""
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None

# Application Flow
col1, col2 = st.columns([2, 3], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("👤 Candidate Academic Profile")
    st.write("Academic metrics used by the Profile Classifier:")
    
    c_branch = st.selectbox("Academic Branch / Major", options=branch_options)
    
    col_cgpa, col_tier = st.columns(2)
    with col_cgpa:
        c_cgpa = st.slider("CGPA", min_value=5.0, max_value=10.0, value=8.0, step=0.1)
    with col_tier:
        c_tier = st.selectbox("College Tier", options=[1, 2, 3], index=1)
        
    st.markdown("#### 📝 Assessments & Performance")
    col_cod, col_apt = st.columns(2)
    with col_cod:
        c_coding = st.slider("Coding Score (0-100)", min_value=0, max_value=100, value=75)
    with col_apt:
        c_aptitude = st.slider("Aptitude Score (0-100)", min_value=0, max_value=100, value=75)
        
    c_comm = st.slider("Communication Score (0-10)", min_value=0, max_value=10, value=8)
    
    st.markdown("#### 💼 Experience & History")
    col_int, col_proj, col_back = st.columns(3)
    with col_int:
        c_internships = st.number_input("Internships Done", min_value=0, max_value=5, value=1)
    with col_proj:
        c_projects = st.number_input("Projects Done", min_value=0, max_value=10, value=2)
    with col_back:
        c_backlogs = st.number_input("Active Backlogs", min_value=0, max_value=5, value=0)
        
    c_dsa = st.checkbox("DSA Skill (Strong Data Structures & Algorithms)", value=True)
    
    st.markdown("#### 📄 Skills & Resume Source")
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF or DOCX)",
        type=["pdf", "docx"],
        help="Upload a resume to automatically detect and extract technical skills."
    )
    
    # Process uploaded file and store extracted skills in session state
    if uploaded_file:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state["last_uploaded_file_key"] != file_key:
            with st.spinner("🔍 Parsing resume and extracting skills..."):
                try:
                    resume_profile = analyze_resume_profile(uploaded_file)
                    st.session_state["extracted_skills_str"] = ", ".join(resume_profile["skills"])
                    st.session_state["resume_text"] = resume_profile["resume_text"]
                    st.session_state["last_uploaded_file_key"] = file_key
                except Exception as e:
                    st.error(f"Error parsing resume: {str(e)}")
    else:
        if st.session_state["last_uploaded_file_key"] is not None:
            st.session_state["last_uploaded_file_key"] = None
            st.session_state["extracted_skills_str"] = ""
            st.session_state["resume_text"] = ""
            
    # Decide value for technical skills text area
    default_skills = st.session_state["extracted_skills_str"]
    if not default_skills and not uploaded_file:
        default_skills = "Python, SQL, Git"
        
    c_skills_input = st.text_area(
        "Technical Skills (separated by commas)",
        value=default_skills,
        placeholder="e.g. HTML, CSS, JavaScript, React, Node.js, SQL",
        help="Skills extracted from your resume will appear here automatically. You can also manually edit, add, or delete skills directly."
    )
    
    analyze_btn = st.button("🚀 Analyze & Predict Career")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if analyze_btn:
        if not c_skills_input.strip():
            st.warning("⚠️ Please enter some technical skills in the text area to analyze.")
            st.session_state["analysis_results"] = None
        else:
            with st.spinner("🔍 Running integrated AI prediction & career mapping..."):
                try:
                    # 1. Parse manual skills from the text area (which now acts as the single source of truth)
                    combined_skills = [s.strip() for s in c_skills_input.split(",") if s.strip()]
                    combined_skills_str = ", ".join(combined_skills)
                    combined_skills_set = {s.lower() for s in combined_skills}
                    resume_text = st.session_state["resume_text"]
                    
                    # 3. Compute skill and resume metrics for the Profile ML model inputs
                    skill_score = round(min(0.5 + (len(combined_skills) * 0.25), 3.5), 2)
                    resume_score = round(min(max(30.0 + (c_cgpa * 4) + (len(combined_skills) * 2) + (c_internships * 5) + (c_projects * 3) - (c_backlogs * 5), 22.5), 130.0), 2)
                    
                    # 4. Predict probabilities using Profile ML Model (Random Forest - 4 classes)
                    ml_probabilities = {}
                    max_ml_confidence = 0.0
                    if ml_model and ml_encoders:
                        try:
                            # Encode branch
                            branch_enc = ml_encoders['branch'].transform([c_branch])[0]
                            
                            input_df = pd.DataFrame([[
                                c_cgpa, branch_enc, c_tier, int(c_dsa), c_coding, c_comm, c_aptitude,
                                c_internships, c_projects, c_backlogs, resume_score, skill_score
                            ]], columns=[
                                'cgpa', 'branch', 'college_tier', 'dsa_skill', 'coding_score',
                                'communication_score', 'aptitude_score', 'internships', 'projects',
                                'backlogs', 'resume_score', 'skill_score'
                            ])
                            
                            probabilities = ml_model.predict_proba(input_df)[0]
                            role_classes = ml_encoders['job_role'].classes_
                            
                            for rc, prob in zip(role_classes, probabilities):
                                ml_probabilities[rc] = prob * 100
                            max_ml_confidence = max(ml_probabilities.values())
                        except Exception as ml_err:
                            st.warning(f"Profile Classifier prediction failed: {str(ml_err)}")
                            
                    # 5. Check skills match against existing database roles first for standard roles
                    from skill_mapper import get_role_skills, save_database
                    db = load_database()
                    
                    db_match_scores = {}
                    for role, data in db.items():
                        req_skills = data.get("required_skills", [])
                        req_set = {s.lower() for s in req_skills}
                        matched_set = req_set & combined_skills_set
                        score = (len(matched_set) / len(req_skills)) * 100 if req_skills else 0
                        db_match_scores[role] = score
                        
                    max_db_match = max(db_match_scores.values()) if db_match_scores else 0.0
                    
                    # 6. Check if Fallback is needed (standard model AI fallback)
                    ai_fallback_triggered = False
                    ai_suggested_role = None
                    ai_suggested_skills = []
                    
                    if max_ml_confidence < 35.0 and max_db_match < 30.0:
                        st.info("🔍 Profile matches no standard categories. Querying AI Advisor for a custom role suggestion...")
                        ai_suggestion = suggest_new_role_ai(resume_text or combined_skills_str, combined_skills)
                        ai_suggested_role = ai_suggestion.get("suggested_role", "Software Engineer").strip()
                        ai_suggested_skills = ai_suggestion.get("required_skills", [])
                        
                        # Save the newly suggested AI role to role_skills.json
                        if ai_suggested_role not in db:
                            db[ai_suggested_role] = {"required_skills": ai_suggested_skills}
                            save_database(db)
                            db = load_database() # Reload
                            
                        # Recalculate matches including the new AI role
                        db_match_scores[ai_suggested_role] = (len({s.lower() for s in ai_suggested_skills} & combined_skills_set) / len(ai_suggested_skills)) * 100 if ai_suggested_skills else 0
                        ai_fallback_triggered = True
                        
                    # 7. Predict specialized roles using Skills ML Model (TF-IDF + 17 classes)
                    # Map 17 specialized skills classes to the 4 academic profile standard classes
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

                    skill_recs = []
                    if skill_model and skill_tfidf and skill_encoder:
                        skills_vector = skill_tfidf.transform([combined_skills_str])
                        probabilities = skill_model.predict_proba(skills_vector)[0]
                        role_classes = skill_encoder.classes_
                        
                        skill_probabilities = {}
                        for rc, prob in zip(role_classes, probabilities):
                            skill_probabilities[rc] = prob * 100
                            
                        for role in role_classes:
                            req_skills = get_role_skills(role)
                            req_set = {s.lower() for s in req_skills}
                            matched_set = req_set & combined_skills_set
                            
                            skill_match_score = (len(matched_set) / len(req_skills)) * 100 if req_skills else 0
                            skills_ml_score = skill_probabilities.get(role, 0.0)
                            
                            # Connect academic features (Placement Predictor probability)
                            mapped_std_role = SPECIALIZED_TO_STANDARD_MAPPING.get(role, "Software Engineer")
                            profile_prob = ml_probabilities.get(mapped_std_role, 0.0) if ml_probabilities else 0.0
                            
                            # Calculate final combined match score: 40% Skills Model Conf + 30% Skill Alignment Gap + 30% Academic profile suitability
                            combined_score = round((0.4 * skills_ml_score) + (0.3 * skill_match_score) + (0.3 * profile_prob), 2)
                            
                            skill_recs.append({
                                "role": role,
                                "score": combined_score,
                                "skills_ml_score": round(skills_ml_score, 2),
                                "skill_score": round(skill_match_score, 2),
                                "profile_score": round(profile_prob, 2),
                                "required_skills": req_skills,
                                "matched_skills": [s for s in req_skills if s.lower() in matched_set],
                                "missing_skills": [s for s in req_skills if s.lower() not in matched_set]
                            })
                        skill_recs = sorted(skill_recs, key=lambda x: x["score"], reverse=True)
                        
                    st.session_state["analysis_results"] = {
                        "combined_skills": combined_skills,
                        "skill_recs": skill_recs,
                        "ml_probabilities": ml_probabilities,
                        "db_match_scores": db_match_scores,
                        "ai_fallback_triggered": ai_fallback_triggered,
                        "ai_suggested_role": ai_suggested_role
                    }
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.session_state["analysis_results"] = None

    if st.session_state["analysis_results"] is not None:
        results = st.session_state["analysis_results"]
        combined_skills = results["combined_skills"]
        skill_recs = results["skill_recs"]
        ml_probabilities = results["ml_probabilities"]
        db_match_scores = results["db_match_scores"]
        ai_fallback_triggered = results["ai_fallback_triggered"]
        ai_suggested_role = results["ai_suggested_role"]

        st.markdown("""
        <div style="margin-top: 10px; margin-bottom: 20px;">
            <h3 style="margin-bottom: 5px; color: #34d399 !important;">✍️ Recommended Career Paths</h3>
            <p style="color: #9ca3af; font-size: 0.9rem; margin-top: 0;">
                Top predicted career paths based on your academic profile, assessment scores, and technical skills (using integrated Random Forest models & TF-IDF mapping).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if skill_recs:
            col_c1, col_c2, col_c3 = st.columns(3)
            cols = [col_c1, col_c2, col_c3]
            
            for idx, rec in enumerate(skill_recs[:3]):
                role_name = rec["role"]
                score = rec["score"]
                sk_ml = rec["skills_ml_score"]
                sk_sc = rec["skill_score"]
                pr_sc = rec["profile_score"]
                
                parent_role = SUB_TO_PARENT_ROLE.get(role_name.upper())
                parent_html = f'<div style="font-size: 0.82rem; color: #a5b4fc; margin-top: -5px; margin-bottom: 8px;">Sub-role of {parent_role}</div>' if parent_role else ""
                
                with cols[idx]:
                    st.markdown(f"""
                    <div class="glass-card" style="border-color: rgba(16, 185, 129, 0.4); min-height: 220px; padding: 20px; margin-bottom: 15px;">
                        <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #34d399; font-weight: 700;">🏆 #{idx+1} Recommendation</span>
                        <h3 style="margin: 10px 0 5px 0; font-size: 1.4rem; color: #6ee7b7 !important;">{role_name}</h3>
                        {parent_html}
                        <p style="color: #e5e7eb; font-size: 0.95rem; margin-bottom: 12px;">Combined Fit: <strong>{score}% Match</strong></p>
                        <p style="color: #9ca3af; font-size: 0.8rem; line-height: 1.4;">
                            Academic Fit: {pr_sc}%<br/>
                            Skill Match: {sk_sc}%<br/>
                            Model Conf: {sk_ml}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander(f"🌐 Why & What Insights for #{idx+1}", expanded=(idx == 0)):
                        with st.spinner("🌐 Fetching live insights..."):
                            try:
                                from tavily_helper import get_career_why_and_what
                                
                                @st.cache_data(ttl=3600)
                                def fetch_why_what_cached(role, skills_tuple):
                                    return get_career_why_and_what(role, list(skills_tuple))
                                    
                                insights = fetch_why_what_cached(role_name, tuple(combined_skills))
                                
                                st.markdown(f"**❓ What it is:**\n{insights.get('what', 'N/A')}")
                                st.markdown(f"**🎯 Why suggest:**\n{insights.get('why', 'N/A')}")
                            except Exception as t_err:
                                st.error(f"Failed to fetch insights: {str(t_err)}")
        else:
            st.info("Career Path Recommender returned no results.")
            
        st.markdown("### 📊 Career Path Match Rankings")
        st.write("Rankings for all career paths predicted using integrated academic profile & technical skills:")
        if skill_recs:
            for idx, rec in enumerate(skill_recs[:10]):
                role = rec["role"]
                score = rec["score"]
                sk_sc = rec["skill_score"]
                pr_sc = rec["profile_score"]
                
                is_top = (idx == 0)
                bar_class = "rec-bar-fill rec-bar-fill-top" if is_top else "rec-bar-fill"
                
                p_role = SUB_TO_PARENT_ROLE.get(role.upper())
                role_display = f"{role} <span style='font-size: 0.85rem; color: #a5b4fc;'>(Sub-role of {p_role})</span>" if p_role else role
                
                st.markdown(f"""
                <div class="rec-item">
                    <div class="rec-label-container">
                        <span>{role_display}</span>
                        <span style="color: #9ca3af; font-size: 0.85rem;">(Academic Fit: {pr_sc}% | Skill Match: {sk_sc}% | Model Conf: {sk_ml}%)</span>
                        <span>{score}% Match</span>
                    </div>
                    <div class="rec-bar-bg">
                        <div class="{bar_class}" style="width: {score}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recommendations found.")
            
        st.markdown("### 🎯 ATS Score & Skills Gap Details")
        
        all_recs_map = {}
        for rec in skill_recs:
            all_recs_map[rec["role"]] = rec
            
        selected_role = st.selectbox(
            "Select any recommended role to view details and ATS gap analysis:",
            options=list(all_recs_map.keys()),
            format_func=lambda x: f"{x} (Sub-role of {SUB_TO_PARENT_ROLE.get(x.upper())})" if SUB_TO_PARENT_ROLE.get(x.upper()) else x
        )
        
        selected_rec = all_recs_map[selected_role]
        ats_score = selected_rec["skill_score"]
        req_skills = selected_rec["required_skills"]
        matched = selected_rec["matched_skills"]
        missing = selected_rec["missing_skills"]
        
        if ats_score >= 75:
            score_class = "score-high"
            score_msg = "Excellent Match"
            score_color = "#10b981"
        elif ats_score >= 50:
            score_class = "score-mid"
            score_msg = "Good Match"
            score_color = "#f59e0b"
        else:
            score_class = "score-low"
            score_msg = "Needs Improvement"
            score_color = "#ef4444"
            
        col_m1, col_m2 = st.columns([1, 2], gap="medium")
        with col_m1:
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <div class="circular-progress">
                    <svg viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="42" stroke="rgba(255, 255, 255, 0.05)" stroke-width="8" fill="transparent" />
                        <circle cx="50" cy="50" r="42" stroke="{score_color}" stroke-width="8" fill="transparent"
                                stroke-dasharray="263.89" stroke-dashoffset="{263.89 * (1 - ats_score/100)}"
                                stroke-linecap="round" style="transform: rotate(-90deg); transform-origin: 50px 50px; transition: stroke-dashoffset 0.5s ease;" />
                    </svg>
                    <div class="circular-progress-text">
                        <span style="font-size: 1.8rem; font-weight: 700; color: #fff;">{ats_score:.0f}%</span>
                        <span style="font-size: 0.7rem; color: #9ca3af; text-transform: uppercase;">Match Score</span>
                    </div>
                </div>
                <div class="{score_class}" style="margin-top: 15px;">
                    {score_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "✨ Skills Gap Analysis", 
                "📋 Required Role Skills", 
                "📝 Your Extracted/Entered Skills",
                "📈 Real-Time Market Insights (Tavily)",
                "💼 Apply to Jobs/Internships",
                "📚 Course Generator (Ollama)"
            ])
            
            with tab1:
                p_role = SUB_TO_PARENT_ROLE.get(selected_role.upper())
                parent_suffix = f" (specialized sub-role under **{p_role}**)" if p_role else ""
                st.write(f"Compare your skills against the requirements for **{selected_role}**{parent_suffix}:")
                
                st.markdown("#### ✅ Matched Technical Skills")
                if matched:
                    badges = "".join([f'<span class="badge badge-matched">{s}</span>' for s in matched])
                    st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)
                else:
                    st.info("No matching skills found for this role.")
                    
                st.markdown("#### ❌ Missing Critical Skills")
                if missing:
                    badges = "".join([f'<span class="badge badge-missing">{s}</span>' for s in missing])
                    st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)
                    st.warning(f"💡 *Actionable Tip: Revise your resume or plan coursework to cover these missing skills.*")
                else:
                    st.success("Great! Your skills list covers all the expectations for this role.")
                    
            with tab2:
                st.write(f"The top technical skills expected for **{selected_role}**:")
                if req_skills:
                    badges = "".join([f'<span class="badge badge-normal">{s}</span>' for s in req_skills])
                    st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)
                else:
                    st.warning("No expected skills list is cached for this role.")
                    
            with tab3:
                st.write("These skills were parsed from your resume or manually input:")
                if combined_skills:
                    badges = "".join([f'<span class="badge badge-normal">{s}</span>' for s in combined_skills])
                    st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)
                else:
                    st.info("No skills are registered.")
                    
            with tab4:
                st.write(f"### 📈 Real-Time Job Market Insights for **{selected_role}**")
                st.write("Fetching live hiring trends, salary ranges, certifications, project ideas, and interview questions directly from current web sources:")
                
                with st.spinner("🌐 Fetching live search data from Tavily..."):
                    try:
                        from tavily_helper import get_job_market_insights
                        
                        @st.cache_data(ttl=3600)
                        def fetch_cached_insights(role):
                            return get_job_market_insights(role)
                            
                        insights = fetch_cached_insights(selected_role)
                        
                        col_sal, col_tr = st.columns(2)
                        with col_sal:
                            st.markdown(f"""
                            <div class="glass-card" style="padding: 20px; border-color: rgba(59, 130, 246, 0.3);">
                                <h5 style="margin-top: 0; color: #60a5fa; margin-bottom: 8px;">💰 Average Compensation</h5>
                                <p style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0;">{insights.get('average_salary', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_tr:
                            st.markdown(f"""
                            <div class="glass-card" style="padding: 20px; border-color: rgba(99, 102, 241, 0.3);">
                                <h5 style="margin-top: 0; color: #818cf8; margin-bottom: 8px;">🚀 Market Hiring Trends</h5>
                                <p style="font-size: 0.9rem; margin-bottom: 0; line-height: 1.4;">{insights.get('market_trends', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        col_comp, col_cert = st.columns(2)
                        with col_comp:
                            st.markdown("#### 🏢 Active Hiring Companies")
                            companies = insights.get("top_companies", [])
                            if companies:
                                for comp in companies:
                                    st.markdown(f"- **{comp}**")
                            else:
                                st.info("No active hiring companies listed.")
                                
                        with col_cert:
                            st.markdown("#### 🎓 Recommended Certifications")
                            certs = insights.get("certifications", [])
                            if certs:
                                for cert in certs:
                                    st.markdown(f"- {cert}")
                            else:
                                st.info("No recommended certifications found.")
                                
                        st.markdown("#### 💡 Recommended Practical Projects")
                        projects = insights.get("project_ideas", [])
                        if projects:
                            for proj in projects:
                                st.markdown(proj if proj.startswith("-") else f"- {proj}")
                        else:
                            st.info("No project recommendations listed.")
                            
                        st.markdown("#### 🗣️ Interview Prep Topics")
                        tips = insights.get("interview_tips", [])
                        if tips:
                            for tip in tips:
                                st.markdown(tip if tip.startswith("-") else f"- {tip}")
                        else:
                            st.info("No interview preparation topics listed.")
                            
                        st.markdown("#### 🗺️ Next Steps Learning Roadmap")
                        st.write(insights.get("learning_roadmap", "Not available."))
                        
                    except Exception as t_err:
                        st.error(f"Failed to fetch market insights: {str(t_err)}")
                        
            with tab5:
                st.write(f"### 💼 Live Jobs & Internships for **{selected_role}**")
                st.write("Browse current openings scraped in real-time from LinkedIn, Naukri.com, Indeed, Internshala, and other hiring portals:")
                
                with st.spinner("🔍 Querying job boards via Tavily..."):
                    try:
                        from tavily_helper import get_active_jobs_and_internships
                        
                        @st.cache_data(ttl=1800)
                        def fetch_live_job_listings(role):
                            return get_active_jobs_and_internships(role)
                            
                        jobs_data = fetch_live_job_listings(selected_role)
                        listings = jobs_data.get("listings", [])
                        
                        if listings:
                            # Count jobs vs internships
                            jobs_count = sum(1 for j in listings if j.get("type", "Job") == "Job")
                            intern_count = sum(1 for j in listings if j.get("type", "") == "Internship")
                            st.markdown(f"Found **{len(listings)}** openings ({jobs_count} Jobs, {intern_count} Internships)")
                            
                            for idx, job in enumerate(listings):
                                title = job.get("title", "N/A")
                                company = job.get("company", "N/A")
                                platform = job.get("platform", "Direct Link")
                                job_type = job.get("type", "Job")
                                url = job.get("url", "#")
                                desc = job.get("description", "")
                                
                                # Color code by type
                                if job_type == "Internship":
                                    type_bg = "rgba(168, 85, 247, 0.15)"
                                    type_color = "#c084fc"
                                    border_color = "#a855f7"
                                else:
                                    type_bg = "rgba(16, 185, 129, 0.15)"
                                    type_color = "#6ee7b7"
                                    border_color = "#10b981"
                                
                                st.markdown(f"""
                                <div class="glass-card" style="padding: 15px; margin-bottom: 12px; border-left: 4px solid {border_color};">
                                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 5px; flex-wrap: wrap; gap: 6px;">
                                        <h5 style="margin: 0; font-size: 1.05rem; color: {type_color} !important;">{title}</h5>
                                        <div style="display: flex; gap: 6px;">
                                            <span style="font-size: 0.7rem; padding: 3px 8px; border-radius: 4px; background: {type_bg}; color: {type_color}; font-weight: 700;">{job_type}</span>
                                            <span style="font-size: 0.7rem; padding: 3px 8px; border-radius: 4px; background: rgba(59, 130, 246, 0.1); color: #60a5fa; font-weight: 600;">{platform}</span>
                                        </div>
                                    </div>
                                    <p style="margin: 4px 0 8px 0; font-size: 0.9rem; color: #e5e7eb;">🏢 <strong>{company}</strong></p>
                                    <p style="margin: 0 0 10px 0; font-size: 0.83rem; color: #9ca3af; line-height: 1.4;">{desc}</p>
                                    <a href="{url}" target="_blank" style="text-decoration: none; display: inline-block; font-size: 0.8rem; font-weight: 700; padding: 6px 14px; border-radius: 6px; color: #064e3b; background-color: #34d399; transition: all 0.2s;">🔗 Apply on {platform}</a>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No active jobs or internships found for this role at the moment. Please try searching again later.")
                    except Exception as jobs_err:
                        st.error(f"Failed to fetch job opportunities: {str(jobs_err)}")
                        
            with tab6:
                st.write("### 📚 AI Course Generator & local Assistant")
                st.write("Generate a personalized week-by-week learning roadmap (via Groq), research documents (via Tavily), and discuss topics with a local Ollama Chatbot.")
                
                # Check health
                @st.cache_resource(ttl=30)
                def get_ollama_status():
                    try:
                        return check_ollama_health()
                    except Exception as e:
                        return {"status": "disconnected", "error": str(e)}

                health_data = get_ollama_status()
                
                if health_data.get("status") != "connected":
                    st.error("❌ Ollama is disconnected. Please make sure Ollama is running locally on port 11434.")
                    st.info("💡 Tip: Start the Ollama application or run `ollama serve` in your terminal. You also need to pull a model, e.g., `ollama pull deepseek-r1:1.5b`.")
                else:
                    st.success("✅ Ollama is connected locally!")
                    
                    # Fetch models
                    @st.cache_data(ttl=60)
                    def get_cached_models():
                        try:
                            return list_models()
                        except Exception:
                            return []

                    local_models = get_cached_models()
                    model_names = [m["name"] for m in local_models] if local_models else []
                    
                    if not model_names:
                        st.warning("⚠️ No local Ollama models found. Please pull a model first.")
                        st.code("ollama pull deepseek-r1:1.5b")
                    else:
                        # Find defaults or deepseek-r1:1.5b
                        default_model_idx = 0
                        for idx, m in enumerate(model_names):
                            if "deepseek-r1:1.5b" in m:
                                default_model_idx = idx
                                break
                            elif "llama3" in m:
                                default_model_idx = idx
                                
                        selected_model = st.selectbox(
                            "Select Local Ollama Model (for Chatbot)",
                            options=model_names,
                            index=default_model_idx
                        )
                        
                        # Goal options
                        st.markdown("#### Choose Learning Goal")
                        
                        # Populate options from selected role or missing skills
                        goal_options = []
                        if selected_role:
                            goal_options.append(f"Learn {selected_role}")
                        if missing:
                            for ms in missing:
                                goal_options.append(f"Master {ms}")
                        goal_options.append("Custom Goal...")
                        
                        goal_choice = st.selectbox(
                            "Select a goal based on your recommendations or input custom",
                            options=goal_options,
                            index=0
                        )
                        
                        if goal_choice == "Custom Goal...":
                            learning_goal = st.text_input("Enter custom learning goal:", value="Learn Python Programming")
                        else:
                            if goal_choice.startswith("Learn "):
                                learning_goal = goal_choice[6:]
                            elif goal_choice.startswith("Master "):
                                learning_goal = goal_choice[7:]
                            else:
                                learning_goal = goal_choice
                            
                        # Course outline state management in session state
                        if "course_goal" not in st.session_state:
                            st.session_state["course_goal"] = ""
                        if "course_outline" not in st.session_state:
                            st.session_state["course_outline"] = None
                        if "course_weeks" not in st.session_state:
                            st.session_state["course_weeks"] = {}
                        if "course_days" not in st.session_state:
                            st.session_state["course_days"] = {}
                            
                        # If the user switches goals, clear previous course state
                        state_key = f"{learning_goal}"
                        if st.session_state.get("course_state_key") != state_key:
                            st.session_state["course_state_key"] = state_key
                            st.session_state["course_outline"] = None
                            st.session_state["course_weeks"] = {}
                            st.session_state["course_days"] = {}
                            if "chat_messages" in st.session_state:
                                del st.session_state["chat_messages"]
                            
                        generate_course_btn = st.button("Generate Course Outline")
                        
                        if generate_course_btn:
                            with st.spinner("Generating weekly course outline using Groq API..."):
                                try:
                                    outline = generate_course_outline(learning_goal)
                                    st.session_state["course_outline"] = outline
                                    st.session_state["course_weeks"] = {}
                                    st.session_state["course_days"] = {}
                                    if "chat_messages" in st.session_state:
                                        del st.session_state["chat_messages"]
                                    st.success("Successfully generated course outline!")
                                    st.rerun()
                                except Exception as gen_err:
                                    st.error(f"Failed to generate course outline: {str(gen_err)}")
                                    
                        outline = st.session_state["course_outline"]
                        if outline:
                            st.markdown(f"### 📖 Course: {outline.get('title', learning_goal)}")
                            st.write(outline.get("description", ""))
                            
                            prereqs = outline.get("prerequisites", [])
                            if prereqs:
                                st.markdown("**📋 Prerequisites & Basics:**")
                                prereqs_badges = "".join([f'<span class="badge badge-normal" style="margin-right: 5px;">{p}</span>' for p in prereqs])
                                st.markdown(f'<div>{prereqs_badges}</div><br>', unsafe_allow_html=True)
                                    
                            st.markdown("---")
                            st.markdown("### 📅 Weekly Syllabus")
                            
                            weeks = outline.get("weeks", [])
                            for w in weeks:
                                w_num = w.get("week")
                                w_title = w.get("title", f"Week {w_num}")
                                w_concepts = w.get("concepts", [])
                                w_focus = w.get("focus", "theory")
                                
                                week_key = f"w_{w_num}"
                                
                                with st.expander(f"Week {w_num}: {w_title} ({w_focus.capitalize()})"):
                                    if w_concepts:
                                        st.write("**Core Concepts:**")
                                        badges = "".join([f'<span class="concept-tag" style="display: inline-block; padding: 4px 10px; border-radius: 4px; background: rgba(99, 102, 241, 0.1); color: #a5b4fc; font-size: 0.8rem; margin: 3px; border: 1px solid rgba(99, 102, 241, 0.2);">{c}</span>' for c in w_concepts])
                                        st.markdown(f'<div>{badges}</div><br>', unsafe_allow_html=True)
                                        
                                    # Check if days breakdown for this week is loaded
                                    week_details = st.session_state["course_weeks"].get(week_key)
                                    
                                    if not week_details:
                                        load_week_btn = st.button(f"Generate Daily Breakdown for Week {w_num}", key=f"btn_w_{w_num}")
                                        if load_week_btn:
                                            with st.spinner(f"Generating daily tasks for Week {w_num} using Groq..."):
                                                try:
                                                    w_data = generate_week_details(
                                                        learning_goal, w_num, w_title, w_concepts
                                                    )
                                                    st.session_state["course_weeks"][week_key] = w_data
                                                    st.rerun()
                                                except Exception as w_err:
                                                    st.error(f"Failed to load week details: {str(w_err)}")
                                    else:
                                        days = week_details.get("days", [])
                                        st.write("**Daily Schedule:**")
                                        for d in days:
                                            d_num = d.get("day")
                                            d_title = d.get("title", f"Day {d_num}")
                                            d_type = d.get("task_type", "theory")
                                            d_duration = d.get("duration_minutes", 60)
                                            d_concepts = d.get("concepts", [])
                                            
                                            day_key = f"d_{w_num}_{d_num}"
                                            
                                            st.markdown(f"**Day {d_num}: {d_title}**")
                                            st.caption(f"⏱ {d_duration} mins | 🏷 Type: {d_type.capitalize()}")
                                            if d_concepts:
                                                st.write("Concepts: " + ", ".join(d_concepts))
                                                
                                            # Lazy load day content
                                            day_content = st.session_state["course_days"].get(day_key)
                                            if not day_content:
                                                load_day_btn = st.button(f"Load Day {d_num} Content", key=f"btn_d_{w_num}_{d_num}")
                                                if load_day_btn:
                                                    with st.spinner(f"Generating details via Groq & searching resources via Tavily..."):
                                                        try:
                                                            d_data = generate_day_details(
                                                                learning_goal, d_title, (w_num - 1) * 7 + d_num, d_type, d_duration
                                                            )
                                                            st.session_state["course_days"][day_key] = d_data
                                                            st.rerun()
                                                        except Exception as d_err:
                                                            st.error(f"Failed to load day content: {str(d_err)}")
                                            else:
                                                st.markdown(f"**Explanation:**\n{day_content.get('description', '')}")
                                                
                                                toc = day_content.get("table_of_contents", [])
                                                if toc:
                                                    st.write("**Topics Covered:**")
                                                    for item in toc:
                                                        st.markdown(f"- {item}")
                                                        
                                                # Resources rendering
                                                resources = day_content.get("resources", [])
                                                if resources:
                                                    st.write("**🔎 Recommended Resources & Tutorials (via Tavily):**")
                                                    for res in resources:
                                                        res_title = res.get("title", "Resource")
                                                        res_url = res.get("url", "#")
                                                        res_source = res.get("source", "web")
                                                        res_desc = res.get("description", "")
                                                        
                                                        if res_source == "youtube":
                                                            icon = "🎥 [YouTube Tutorial]"
                                                        elif res_source == "research_paper":
                                                            icon = "🎓 [Research Paper]"
                                                        else:
                                                            icon = "📖 [Official Documentation]"
                                                            
                                                        st.markdown(f"- **{icon} [{res_title}]({res_url})**")
                                                        if res_desc:
                                                            st.markdown(f"  *{res_desc}*")
                                                            
                                            st.markdown("---")
                                            
                            # ─── Chatbot Section ───────────────────
                            st.markdown("---")
                            st.markdown("### 💬 Course Chatbot Assistant (Ollama)")
                            st.write(f"Ask the chatbot questions about the **{outline.get('title')}** course. Responses are processed locally using `{selected_model}`.")
                            
                            # Initialize chatbot message history
                            if "chat_messages" not in st.session_state:
                                st.session_state["chat_messages"] = []
                                
                            # Display messages
                            for msg in st.session_state["chat_messages"]:
                                with st.chat_message(msg["role"]):
                                    st.markdown(msg["content"])
                                    
                            # Input
                            if user_chat_input := st.chat_input("Type your question here..."):
                                # Render user message
                                with st.chat_message("user"):
                                    st.markdown(user_chat_input)
                                st.session_state["chat_messages"].append({"role": "user", "content": user_chat_input})
                                
                                # Send query to Ollama
                                with st.spinner("AI Tutor is thinking..."):
                                    try:
                                        chat_payload = [
                                            {"role": "system", "content": "You are a professional educational tutor. Provide helpful and deep explanations to students based on their course details."},
                                            {"role": "system", "content": f"The student is taking a course titled '{outline.get('title')}' with goal '{learning_goal}'. Description: '{outline.get('description')}'."},
                                        ]
                                        # Append last 8 messages
                                        for msg in st.session_state["chat_messages"][-8:]:
                                            chat_payload.append({"role": msg["role"], "content": msg["content"]})
                                            
                                        assistant_response = call_ollama_chat(selected_model, chat_payload)
                                        
                                        # Render response
                                        with st.chat_message("assistant"):
                                            st.markdown(assistant_response)
                                        st.session_state["chat_messages"].append({"role": "assistant", "content": assistant_response})
                                        st.rerun()
                                    except Exception as chat_err:
                                        st.error(f"Chatbot failed: {str(chat_err)}")
    else:
        # Default placeholder container with rich instructions
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 70px 40px;">
            <div style="font-size: 4.5rem; margin-bottom: 24px;">🤖</div>
            <h3>Waiting for Career Analysis Inputs...</h3>
            <p style="color: #9ca3af; max-width: 500px; margin: 0 auto 24px auto;">
                Provide your academic parameters and either upload your resume (PDF/DOCX) or enter technical skills on the left panel, then click the Analyze button. The hybrid engine will run both Profile-based and Skills-based Random Forest models to recommend your ideal careers.
            </p>
        </div>
        """, unsafe_allow_html=True)
