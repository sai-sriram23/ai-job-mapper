# -*- coding: utf-8 -*-
import streamlit as st
import json
import streamlit.components.v1 as components
from tavily_helper import get_active_jobs_and_internships
from career_accelerator import (
    generate_complete_resume_data,
    generate_ats_resume_bullets,
    evaluate_job_resume_fit
)
from resume_builder_helper import (
    render_resume_html_ats,
    render_resume_html_modern,
    render_resume_plain_text,
    calculate_resume_ats_score
)

def render_automated_resume_builder_view():
    """
    Renders the Full-Featured Automated Resume Builder with AI STAR bullets, 
    live template preview, ATS scoring, and 1-click HTML download.
    """
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 📄 AI Automated Resume Builder")
    st.write("Construct an ATS-optimized, professional resume with AI-generated STAR bullet points, live template rendering, and instant export.")
    st.markdown('</div>', unsafe_allow_html=True)

    consensus = st.session_state.get("consensus_profile")
    if consensus:
        c_role = consensus.get("top_consensus_role", "")
        c_score = consensus.get("consensus_score", 0)
        st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid #6366f1; padding: 12px 18px; border-radius: 12px; margin: 16px 0;">
            🧠 <strong>Dual ML + LangChain Consensus Sync:</strong> Resume targeted for <strong>{c_role}</strong> ({c_score}% Combined Fit).
        </div>
        """, unsafe_allow_html=True)

    # Initialize resume data in session state if missing
    if "resume_builder_state" not in st.session_state:
        # Pre-fill skills from main recommender if available
        extracted_skills = st.session_state.get("extracted_skills_str", "")
        skills_list = [s.strip() for s in extracted_skills.split(",") if s.strip()] if extracted_skills else []

        st.session_state["resume_builder_state"] = {
            "personal_info": {
                "name": "",
                "title": "",
                "email": "",
                "phone": "",
                "location": "",
                "linkedin": "",
                "github": ""
            },
            "professional_summary": "",
            "target_role": "",
            "skills": {
                "Languages & Core": skills_list[:3] if len(skills_list) >= 3 else skills_list,
                "Frameworks & Tools": skills_list[3:] if len(skills_list) > 3 else [],
                "Databases & Cloud": [],
                "Soft Skills": []
            },
            "experiences": [],
            "education": [],
            "certifications": []
        }

    res = st.session_state["resume_builder_state"]
    if "personal_info" not in res or not isinstance(res.get("personal_info"), dict):
        res["personal_info"] = {
            "name": "", "title": "", "email": "", "phone": "",
            "location": "", "linkedin": "", "github": ""
        }
    if "skills" not in res or not isinstance(res.get("skills"), dict):
        res["skills"] = {}
    if "experiences" not in res or not isinstance(res.get("experiences"), list):
        res["experiences"] = []
    if "education" not in res or not isinstance(res.get("education"), list):
        res["education"] = []
    if "certifications" not in res or not isinstance(res.get("certifications"), list):
        res["certifications"] = []

    col_edit, col_prev = st.columns([1, 1], gap="large")

    with col_edit:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ✏️ Resume Content Editor")

        # Tabbed form controls
        tab_p, tab_sum, tab_sk, tab_exp, tab_edu = st.tabs([
            "👤 Contact", "📝 Summary", "🛠️ Skills", "💼 Experience", "🎓 Education"
        ])

        with tab_p:
            st.markdown("#### Personal Contact Information")
            res["personal_info"]["name"] = st.text_input("Full Name", value=res["personal_info"].get("name", ""))
            res["personal_info"]["title"] = st.text_input("Target Professional Title", value=res["personal_info"].get("title", ""))
            res["target_role"] = res["personal_info"]["title"]

            c1, c2 = st.columns(2)
            with c1:
                res["personal_info"]["email"] = st.text_input("Email Address", value=res["personal_info"].get("email", ""))
                res["personal_info"]["location"] = st.text_input("City, State / Remote", value=res["personal_info"].get("location", ""))
            with c2:
                res["personal_info"]["phone"] = st.text_input("Phone Number", value=res["personal_info"].get("phone", ""))
                res["personal_info"]["linkedin"] = st.text_input("LinkedIn Profile URL", value=res["personal_info"].get("linkedin", ""))
            res["personal_info"]["github"] = st.text_input("GitHub / Portfolio Link", value=res["personal_info"].get("github", ""))

        with tab_sum:
            st.markdown("#### Professional Executive Summary")
            
            if st.button("✨ AI Auto-Generate Summary (Groq)", key="btn_gen_sum"):
                with st.spinner("Generating ATS executive summary..."):
                    all_skills = []
                    for v in res["skills"].values():
                        if isinstance(v, list): all_skills.extend(v)
                    comp_data = generate_complete_resume_data(res["personal_info"], res["experiences"], res["education"], all_skills, res["target_role"])
                    res["professional_summary"] = comp_data.get("professional_summary", res["professional_summary"])
                    st.success("Summary generated!")
                    st.rerun()

            res["professional_summary"] = st.text_area(
                "Executive Summary Text",
                value=res.get("professional_summary", ""),
                height=120,
                help="A compelling 2-3 line summary highlighting your key skills and achievements."
            )

        with tab_sk:
            st.markdown("#### Categorized Technical Skills")
            for category in ["Languages & Core", "Frameworks & Tools", "Databases & Cloud", "Soft Skills"]:
                existing_list = res["skills"].get(category, [])
                val_str = ", ".join(existing_list) if isinstance(existing_list, list) else str(existing_list)
                new_str = st.text_input(f"{category} (comma-separated)", value=val_str)
                res["skills"][category] = [s.strip() for s in new_str.split(",") if s.strip()]

        with tab_exp:
            st.markdown("#### Work Experience & Key Projects")
            
            # Button to add new experience
            if st.button("➕ Add Experience / Project Item", key="btn_add_exp"):
                res["experiences"].append({
                    "title": "",
                    "company": "",
                    "period": "",
                    "location": "",
                    "bullets": []
                })
                st.rerun()

            if not res["experiences"]:
                st.info("💡 No experience entries added yet. Click **'➕ Add Experience / Project Item'** to create your first entry.")

            for idx, exp in enumerate(res["experiences"]):
                display_title = exp.get('title') or f"Position #{idx+1}"
                display_comp = exp.get('company') or "Organization"
                with st.expander(f"📌 {display_title} @ {display_comp}", expanded=(idx==0)):
                    exp["title"] = st.text_input(f"Title / Role #{idx+1}", value=exp.get("title", ""), key=f"exp_title_{idx}")
                    exp["company"] = st.text_input(f"Company / Organization #{idx+1}", value=exp.get("company", ""), key=f"exp_comp_{idx}")
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        exp["period"] = st.text_input(f"Period #{idx+1}", value=exp.get("period", ""), key=f"exp_per_{idx}")
                    with col_p2:
                        exp["location"] = st.text_input(f"Location #{idx+1}", value=exp.get("location", ""), key=f"exp_loc_{idx}")

                    bullets_str = "\n".join(exp.get("bullets", []))
                    new_bullets = st.text_area(f"STAR Bullets (one per line) #{idx+1}", value=bullets_str, height=120, key=f"exp_bul_{idx}")
                    exp["bullets"] = [b.strip() for b in new_bullets.split("\n") if b.strip()]

                    if st.button(f"⚡ Generate STAR Bullets via AI #{idx+1}", key=f"btn_star_{idx}"):
                        with st.spinner("Generating quantified STAR bullets..."):
                            all_skills = [s for sub in res["skills"].values() for s in sub]
                            ai_res = generate_ats_resume_bullets(exp["title"] or "Software Engineer", all_skills[:3], [], all_skills)
                            ai_bullets = [bp.get("star_bullet") for bp in ai_res.get("bullet_points", []) if bp.get("star_bullet")]
                            if ai_bullets:
                                exp["bullets"] = ai_bullets
                                st.success("STAR bullets generated!")
                                st.rerun()

                    if st.button(f"🗑️ Remove Entry #{idx+1}", key=f"btn_del_exp_{idx}"):
                        res["experiences"].pop(idx)
                        st.rerun()

        with tab_edu:
            st.markdown("#### Education & Academic Qualifications")
            
            if st.button("➕ Add Education Item", key="btn_add_edu"):
                res["education"].append({
                    "degree": "",
                    "institution": "",
                    "year": "",
                    "cgpa": ""
                })
                st.rerun()

            if not res["education"]:
                st.info("💡 No education items added yet. Click **'➕ Add Education Item'** to add your degree details.")

            for idx, edu in enumerate(res["education"]):
                display_deg = edu.get('degree') or f"Degree #{idx+1}"
                with st.expander(f"🎓 {display_deg}", expanded=(idx==0)):
                    edu["degree"] = st.text_input(f"Degree / Qualification #{idx+1}", value=edu.get("degree", ""), key=f"edu_deg_{idx}")
                    edu["institution"] = st.text_input(f"College / Institution #{idx+1}", value=edu.get("institution", ""), key=f"edu_inst_{idx}")
                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        edu["year"] = st.text_input(f"Graduation Year #{idx+1}", value=edu.get("year", ""), key=f"edu_yr_{idx}")
                    with col_e2:
                        edu["cgpa"] = st.text_input(f"CGPA / Percentage #{idx+1}", value=edu.get("cgpa", ""), key=f"edu_cgpa_{idx}")

                    if st.button(f"🗑️ Remove Education Entry #{idx+1}", key=f"btn_del_edu_{idx}"):
                        res["education"].pop(idx)
                        st.rerun()

            st.markdown("#### Certifications")
            certs_str = "\n".join(res.get("certifications", []))
            new_certs = st.text_area("Certifications (one per line)", value=certs_str, height=80, placeholder="e.g. AWS Certified Solutions Architect")
            res["certifications"] = [c.strip() for c in new_certs.split("\n") if c.strip()]

        st.markdown('</div>', unsafe_allow_html=True)

    with col_prev:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 👁️ Live Resume Preview & Export")

        # Template Selection
        template_choice = st.radio(
            "Select Resume Template:",
            options=["✨ Modern Glassmorphic", "📄 ATS Standard Printable", "📋 Plain Text Copy Mode"],
            horizontal=True
        )

        # Calculate ATS Score
        ats_eval = calculate_resume_ats_score(res, res.get("target_role", "Software Engineer"))
        score_val = ats_eval.get("score", 85)

        # ATS Score Badge
        score_color = "#10b981" if score_val >= 80 else "#f59e0b"
        st.markdown(f"""
        <div style="background: rgba(17,24,39,0.8); border: 1.5px solid {score_color}; padding: 12px 18px; border-radius: 12px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong style="color: {score_color}; font-size: 1.1rem;">ATS Resume Score: {score_val}/100</strong>
                <div style="font-size: 0.8rem; color: #9ca3af;">Optimized for ATS resume scanners</div>
            </div>
            <div style="background: {score_color}; color: #000; font-weight: 800; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">
                {'EXCELLENT' if score_val>=80 else 'GOOD'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if ats_eval.get("feedback"):
            with st.expander("💡 ATS Optimization Checklist"):
                for fb in ats_eval["feedback"]:
                    st.markdown(f"- {fb}")

        # Render HTML based on selection
        if template_choice == "✨ Modern Glassmorphic":
            html_content = render_resume_html_modern(res)
            components.html(html_content, height=650, scrolling=True)
        elif template_choice == "📄 ATS Standard Printable":
            html_content = render_resume_html_ats(res)
            components.html(html_content, height=650, scrolling=True)
        else:
            text_content = render_resume_plain_text(res)
            st.code(text_content, language="text")
            html_content = render_resume_html_ats(res)

        st.markdown("---")
        # Export Actions
        col_dl, col_copy = st.columns(2)
        with col_dl:
            st.download_button(
                label="⬇️ Download HTML Resume File",
                data=html_content,
                file_name=f"{res['personal_info'].get('name', 'Resume').replace(' ', '_')}_Resume.html",
                mime="text/html",
                key="btn_download_resume"
            )
        with col_copy:
            st.info("💡 *Tip: Open the downloaded .html file in Chrome/Edge and press Ctrl+P to save as crisp PDF!*")

        st.markdown('</div>', unsafe_allow_html=True)


def render_live_job_portal_view():
    """
    Renders the Standalone Live Job & Internship Portal with real-time search, 
    resume match evaluation, and saved jobs bookmark manager.
    """
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 💼 Live Job & Internship Portal")
    st.write("Search current open positions in real-time across LinkedIn, Naukri, Indeed, Internshala, and Wellfound with 1-click AI Resume Match Evaluation.")
    st.markdown('</div>', unsafe_allow_html=True)

    consensus = st.session_state.get("consensus_profile")
    if consensus:
        c_role = consensus.get("top_consensus_role", "")
        c_score = consensus.get("consensus_score", 0)
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; padding: 12px 18px; border-radius: 12px; margin: 16px 0;">
            🧠 <strong>Dual ML + LangChain Consensus Sync:</strong> Live hiring portal auto-synced for <strong>{c_role}</strong> ({c_score}% Combined Fit).
        </div>
        """, unsafe_allow_html=True)

    # Initialize state for saved jobs
    if "saved_jobs" not in st.session_state:
        st.session_state["saved_jobs"] = []

    # Search bar & Filters
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Search Hiring Opportunities")
    
    col_q, col_loc, col_type = st.columns([2, 1, 1])

    default_role = st.session_state.get("selected_role", "Software Engineer")
    with col_q:
        search_query = st.text_input("Job Role / Title / Keywords", value=default_role)
    with col_loc:
        location_filter = st.selectbox("Location", options=["All", "Remote", "India", "USA", "Europe", "Singapore"])
    with col_type:
        type_filter = st.selectbox("Position Type", options=["All", "Job", "Internship"])

    fetch_btn = st.button("🚀 Search Live Openings (Tavily Real-Time Web Search)", key="btn_search_jobs")
    st.markdown('</div>', unsafe_allow_html=True)

    # Cache job search results
    if "job_portal_listings" not in st.session_state or fetch_btn:
        with st.spinner(f"Querying top hiring portals for '{search_query}'..."):
            try:
                jobs_data = get_active_jobs_and_internships(search_query, location=location_filter, job_type=type_filter)
                st.session_state["job_portal_listings"] = jobs_data.get("listings", [])
            except Exception as err:
                st.error(f"Job search failed: {str(err)}")
                st.session_state["job_portal_listings"] = []

    listings = st.session_state.get("job_portal_listings", [])

    tab_list, tab_saved = st.tabs([
        f"💼 Open Listings ({len(listings)})", 
        f"⭐ Saved Jobs ({len(st.session_state['saved_jobs'])})"
    ])

    with tab_list:
        if listings:
            st.markdown(f"Found **{len(listings)}** verified active hiring opportunities:")
            
            # User candidate skills for fit matching
            extracted_skills = st.session_state.get("extracted_skills_str", "Python, SQL, Git")
            user_skills = [s.strip() for s in extracted_skills.split(",") if s.strip()]
            resume_text = st.session_state.get("resume_text", "")

            for idx, job in enumerate(listings):
                j_title = job.get("title", "N/A")
                j_company = job.get("company", "N/A")
                j_platform = job.get("platform", "Direct Link")
                j_type = job.get("type", "Job")
                j_url = job.get("url", "#")
                j_desc = job.get("description", "")

                border_color = "#a855f7" if j_type == "Internship" else "#10b981"
                type_bg = "rgba(168, 85, 247, 0.15)" if j_type == "Internship" else "rgba(16, 185, 129, 0.15)"
                type_color = "#c084fc" if j_type == "Internship" else "#6ee7b7"

                st.markdown(f"""
                <div class="glass-card" style="padding: 20px; margin-bottom: 16px; border-left: 4px solid {border_color};">
                    <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <h4 style="margin: 0 0 4px 0; color: {type_color} !important; font-size: 1.15rem;">{j_title}</h4>
                            <div style="color: #e5e7eb; font-size: 0.95rem; font-weight: 600;">🏢 {j_company}</div>
                        </div>
                        <div style="display: flex; gap: 6px;">
                            <span style="font-size: 0.75rem; padding: 4px 10px; border-radius: 6px; background: {type_bg}; color: {type_color}; font-weight: 700;">{j_type}</span>
                            <span style="font-size: 0.75rem; padding: 4px 10px; border-radius: 6px; background: rgba(59, 130, 246, 0.15); color: #60a5fa; font-weight: 600;">{j_platform}</span>
                        </div>
                    </div>
                    <p style="margin: 10px 0 14px 0; font-size: 0.88rem; color: #9ca3af; line-height: 1.5;">{j_desc}</p>
                    <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
                        <a href="{j_url}" target="_blank" style="text-decoration: none; font-size: 0.85rem; font-weight: 700; padding: 8px 18px; border-radius: 8px; color: #064e3b; background-color: #34d399;">🔗 Apply on {j_platform}</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_fit, col_save = st.columns([1, 1])
                with col_fit:
                    fit_key = f"fit_eval_{idx}_{j_title[:10]}"
                    if st.button(f"🎯 Check Resume Fit", key=f"btn_fit_{idx}"):
                        with st.spinner("Analyzing candidate suitability..."):
                            eval_res = evaluate_job_resume_fit(j_title, j_company, j_desc, user_skills, resume_text)
                            st.session_state[fit_key] = eval_res

                    fit_data = st.session_state.get(fit_key)
                    if fit_data:
                        score = fit_data.get("match_score", 75)
                        badge_c = "#10b981" if score >= 75 else "#f59e0b"
                        st.markdown(f"""
                        <div style="background: rgba(17,24,39,0.9); border: 1px solid {badge_c}; padding: 12px; border-radius: 8px; margin-top: 8px;">
                            <strong style="color: {badge_c};">Fit Score: {score}% — {fit_data.get('fit_rating', '')}</strong>
                            <p style="margin: 4px 0; font-size: 0.83rem; color: #e5e7eb;">{fit_data.get('verdict_summary', '')}</p>
                            <div style="font-size: 0.78rem; color: #a5b4fc;">💡 {fit_data.get('application_tip', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)

                with col_save:
                    is_saved = any(s.get("url") == j_url for s in st.session_state["saved_jobs"])
                    if is_saved:
                        st.success("⭐ Saved to Bookmarks")
                    else:
                        if st.button(f"⭐ Bookmark Job", key=f"btn_save_{idx}"):
                            st.session_state["saved_jobs"].append(job)
                            st.success("Job bookmarked!")
                            st.rerun()
        else:
            st.info("No active openings found for this search. Try broadening your role title or location filters.")

    with tab_saved:
        saved = st.session_state.get("saved_jobs", [])
        if saved:
            st.markdown(f"### ⭐ Your Bookmarked Jobs ({len(saved)})")
            for idx, sj in enumerate(saved):
                st.markdown(f"""
                <div class="glass-card" style="padding: 15px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <h5 style="margin: 0; color: #34d399 !important;">{sj.get('title')}</h5>
                        <span style="font-size: 0.75rem; background: rgba(59, 130, 246, 0.2); color: #60a5fa; padding: 2px 6px; border-radius: 4px;">{sj.get('platform')}</span>
                    </div>
                    <p style="margin: 4px 0; font-size: 0.85rem; color: #e5e7eb;">🏢 {sj.get('company')}</p>
                    <a href="{sj.get('url')}" target="_blank" style="color: #6ee7b7; font-size: 0.82rem; font-weight: 700;">🔗 Open Application Link</a>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"🗑️ Remove Bookmark #{idx+1}", key=f"btn_rm_save_{idx}"):
                    st.session_state["saved_jobs"].pop(idx)
                    st.rerun()
        else:
            st.info("No saved jobs yet. Click '⭐ Bookmark Job' on any listing to save it here.")
