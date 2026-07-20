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
                
                with cols[idx]:
                    st.markdown(f"""
                    <div class="glass-card" style="border-color: rgba(16, 185, 129, 0.4); min-height: 220px; padding: 20px; margin-bottom: 15px;">
                        <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #34d399; font-weight: 700;">🏆 #{idx+1} Recommendation</span>
                        <h3 style="margin: 10px 0 5px 0; font-size: 1.4rem; color: #6ee7b7 !important;">{role_name}</h3>
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
                sk_ml = rec["skills_ml_score"]
                sk_sc = rec["skill_score"]
                pr_sc = rec["profile_score"]
                
                is_top = (idx == 0)
                bar_class = "rec-bar-fill rec-bar-fill-top" if is_top else "rec-bar-fill"
                
                st.markdown(f"""
                <div class="rec-item">
                    <div class="rec-label-container">
                        <span>{role}</span>
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
            options=list(all_recs_map.keys())
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
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "✨ Skills Gap Analysis", 
                "📋 Required Role Skills", 
                "📝 Your Extracted/Entered Skills",
                "📈 Real-Time Market Insights (Tavily)",
                "💼 Apply to Jobs/Internships"
            ])
            
            with tab1:
                st.write(f"Compare your skills against the requirements for **{selected_role}**:")
                
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
