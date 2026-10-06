# -*- coding: utf-8 -*-
import json

def render_resume_html_ats(data: dict) -> str:
    """
    Generates a clean, single-column ATS-friendly HTML resume optimized for ATS parsers.
    """
    p = data.get("personal_info", {})
    summary = data.get("professional_summary", "")
    skills = data.get("skills", {})
    experiences = data.get("experiences", [])
    education = data.get("education", [])
    certs = data.get("certifications", [])

    skills_html = ""
    if isinstance(skills, dict):
        for cat, item_list in skills.items():
            if item_list:
                items_str = ", ".join(item_list) if isinstance(item_list, list) else str(item_list)
                skills_html += f"<p style='margin: 4px 0;'><strong>{cat}:</strong> {items_str}</p>"
    elif isinstance(skills, list):
        skills_html = f"<p style='margin: 4px 0;'>{', '.join(skills)}</p>"

    exp_html = ""
    for exp in experiences:
        bullets = exp.get("bullets", [])
        bullets_items = "".join([f"<li style='margin-bottom: 4px;'>{b}</li>" for b in bullets])
        exp_html += f"""
        <div style="margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; font-weight: bold;">
                <span>{exp.get('title', '')} &mdash; {exp.get('company', '')}</span>
                <span>{exp.get('period', '')}</span>
            </div>
            <div style="font-style: italic; font-size: 0.9em; margin-bottom: 4px; color: #555;">{exp.get('location', '')}</div>
            <ul style="margin: 4px 0 8px 20px; padding: 0;">
                {bullets_items}
            </ul>
        </div>
        """

    edu_html = ""
    for edu in education:
        edu_html += f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <div>
                <strong>{edu.get('degree', '')}</strong> &mdash; {edu.get('institution', '')}
            </div>
            <div>{edu.get('year', '')} {f"| CGPA: {edu.get('cgpa', '')}" if edu.get('cgpa') else ""}</div>
        </div>
        """

    cert_html = ""
    if certs:
        cert_items = "".join([f"<li>{c}</li>" for c in certs])
        cert_html = f"<ul style='margin: 4px 0 0 20px; padding: 0;'>{cert_items}</ul>"

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{p.get('name', 'Resume')} - ATS Resume</title>
    <style>
        body {{ font-family: Arial, Helvetica, sans-serif; color: #111; line-height: 1.4; margin: 30px; background: #fff; }}
        h1 {{ margin: 0 0 4px 0; font-size: 22pt; text-transform: uppercase; letter-spacing: 1px; color: #000; text-align: center; }}
        .contact {{ text-align: center; font-size: 9.5pt; margin-bottom: 18px; color: #333; }}
        .contact span {{ margin: 0 6px; }}
        h2 {{ font-size: 11pt; text-transform: uppercase; border-bottom: 1.5px solid #111; padding-bottom: 2px; margin: 16px 0 8px 0; letter-spacing: 0.5px; color: #000; }}
        p {{ margin: 4px 0; font-size: 10pt; }}
        ul {{ font-size: 10pt; }}
    </style>
</head>
<body>
    <h1>{p.get('name', 'CANDIDATE NAME')}</h1>
    <div class="contact">
        <span>{p.get('location', '')}</span> |
        <span>{p.get('phone', '')}</span> |
        <span>{p.get('email', '')}</span> |
        <span><a href="{p.get('linkedin', '#')}">LinkedIn</a></span> |
        <span><a href="{p.get('github', '#')}">GitHub</a></span>
    </div>

    {'<h2>PROFESSIONAL SUMMARY</h2><p>' + summary + '</p>' if summary else ''}

    <h2>TECHNICAL SKILLS</h2>
    {skills_html}

    {'<h2>WORK EXPERIENCE & PROJECTS</h2>' + exp_html if exp_html else ''}

    {'<h2>EDUCATION</h2>' + edu_html if edu_html else ''}

    {'<h2>CERTIFICATIONS & ACHIEVEMENTS</h2>' + cert_html if cert_html else ''}
</body>
</html>
    """
    return html


def render_resume_html_modern(data: dict) -> str:
    """
    Generates a high-visual-impact modern dark glassmorphic HTML resume.
    """
    p = data.get("personal_info", {})
    summary = data.get("professional_summary", "")
    skills = data.get("skills", {})
    experiences = data.get("experiences", [])
    education = data.get("education", [])
    certs = data.get("certifications", [])

    skills_badges = ""
    if isinstance(skills, dict):
        for cat, item_list in skills.items():
            if item_list:
                items = item_list if isinstance(item_list, list) else [str(item_list)]
                badges = "".join([f'<span style="background: rgba(99,102,241,0.2); color:#a5b4fc; padding:4px 10px; border-radius:12px; font-size:0.8rem; margin-right:6px; margin-bottom:6px; display:inline-block; border:1px solid rgba(99,102,241,0.3);">{s}</span>' for s in items])
                skills_badges += f'<div style="margin-bottom:10px;"><strong style="color:#818cf8; display:block; margin-bottom:4px;">{cat}</strong><div>{badges}</div></div>'

    exp_html = ""
    for exp in experiences:
        bullets = exp.get("bullets", [])
        bullets_items = "".join([f'<li style="margin-bottom: 6px; color: #e5e7eb;">{b}</li>' for b in bullets])
        exp_html += f"""
        <div style="background: rgba(255,255,255,0.03); border-left: 3px solid #34d399; padding: 14px 18px; border-radius: 8px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                <h4 style="margin:0; color:#34d399; font-size:1.1rem;">{exp.get('title', '')} <span style="color:#9ca3af; font-weight:normal; font-size:0.9rem;">at {exp.get('company', '')}</span></h4>
                <span style="font-size:0.82rem; color:#a5b4fc; background:rgba(165,180,252,0.1); padding:2px 8px; border-radius:4px;">{exp.get('period', '')}</span>
            </div>
            <ul style="margin: 8px 0 0 16px; padding: 0; font-size: 0.92rem;">
                {bullets_items}
            </ul>
        </div>
        """

    edu_html = ""
    for edu in education:
        edu_html += f"""
        <div style="background: rgba(255,255,255,0.03); padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; display: flex; justify-content: space-between;">
            <div>
                <strong style="color: #60a5fa;">{edu.get('degree', '')}</strong>
                <div style="font-size:0.85rem; color: #9ca3af;">{edu.get('institution', '')}</div>
            </div>
            <div style="text-align: right; font-size:0.85rem; color: #a5b4fc;">
                <div>{edu.get('year', '')}</div>
                {f'<div style="color:#34d399;">CGPA: {edu.get("cgpa")}</div>' if edu.get("cgpa") else ''}
            </div>
        </div>
        """

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{p.get('name', 'Resume')} - Modern Tech Resume</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; line-height: 1.5; }}
        .header {{ text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 20px; margin-bottom: 24px; }}
        .name {{ font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #a5b4fc, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }}
        .title {{ color: #34d399; font-size: 1.1rem; font-weight: 600; margin-top: 4px; }}
        .contact {{ font-size: 0.85rem; color: #9ca3af; margin-top: 10px; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; }}
        .section-title {{ color: #818cf8; font-size: 1.15rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid rgba(129, 140, 248, 0.3); padding-bottom: 4px; margin: 24px 0 14px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 class="name">{p.get('name', 'CANDIDATE NAME')}</h1>
        <div class="title">{p.get('title', 'Software Engineer')}</div>
        <div class="contact">
            <span>📍 {p.get('location', '')}</span>
            <span>📞 {p.get('phone', '')}</span>
            <span>✉️ {p.get('email', '')}</span>
            <span>🔗 {p.get('linkedin', '')}</span>
            <span>💻 {p.get('github', '')}</span>
        </div>
    </div>

    {'<div class="section-title">Professional Summary</div><p style="color:#cbd5e1; font-size:0.95rem; background:rgba(255,255,255,0.02); padding:12px; border-radius:8px;">' + summary + '</p>' if summary else ''}

    <div class="section-title">Technical Skills</div>
    {skills_badges}

    {'<div class="section-title">Experience & Projects</div>' + exp_html if exp_html else ''}

    {'<div class="section-title">Education</div>' + edu_html if edu_html else ''}
</body>
</html>
    """
    return html


def render_resume_plain_text(data: dict) -> str:
    """Generates plain text resume string for quick copy pasting."""
    p = data.get("personal_info", {})
    summary = data.get("professional_summary", "")
    skills = data.get("skills", {})
    experiences = data.get("experiences", [])
    education = data.get("education", [])

    lines = []
    lines.append(p.get("name", "CANDIDATE NAME").upper())
    lines.append(f"{p.get('location', '')} | {p.get('phone', '')} | {p.get('email', '')}")
    lines.append(f"LinkedIn: {p.get('linkedin', '')} | GitHub: {p.get('github', '')}")
    lines.append("=" * 60)

    if summary:
        lines.append("\nPROFESSIONAL SUMMARY")
        lines.append("-" * 30)
        lines.append(summary)

    lines.append("\nTECHNICAL SKILLS")
    lines.append("-" * 30)
    if isinstance(skills, dict):
        for cat, item_list in skills.items():
            items_str = ", ".join(item_list) if isinstance(item_list, list) else str(item_list)
            lines.append(f"{cat}: {items_str}")
    else:
        lines.append(str(skills))

    if experiences:
        lines.append("\nEXPERIENCE & PROJECTS")
        lines.append("-" * 30)
        for exp in experiences:
            lines.append(f"{exp.get('title', '')} - {exp.get('company', '')} ({exp.get('period', '')})")
            for b in exp.get("bullets", []):
                lines.append(f"  * {b}")
            lines.append("")

    if education:
        lines.append("\nEDUCATION")
        lines.append("-" * 30)
        for edu in education:
            lines.append(f"{edu.get('degree', '')}, {edu.get('institution', '')} ({edu.get('year', '')}) CGPA: {edu.get('cgpa', 'N/A')}")

    return "\n".join(lines)


def calculate_resume_ats_score(data: dict, target_role: str) -> dict:
    """
    Calculates ATS optimization score based on key resume criteria.
    """
    p = data.get("personal_info", {})
    summary = data.get("professional_summary", "")
    skills = data.get("skills", {})
    experiences = data.get("experiences", [])

    score = 40
    feedback = []

    if p.get("name") and p.get("email") and p.get("phone"):
        score += 15
    else:
        feedback.append("Add complete contact info (Name, Email, Phone)")

    if summary and len(summary) > 50:
        score += 15
    else:
        feedback.append("Add a 2-3 sentence executive summary")

    total_skills = 0
    if isinstance(skills, dict):
        for v in skills.values():
            total_skills += len(v) if isinstance(v, list) else 1
    if total_skills >= 5:
        score += 15
    else:
        feedback.append("Add at least 5-8 relevant technical skills")

    has_metrics = False
    if experiences:
        score += 15
        for exp in experiences:
            for b in exp.get("bullets", []):
                if any(char.isdigit() for char in b) or "%" in b:
                    has_metrics = True
                    break

    if has_metrics:
        score += 10
    else:
        feedback.append("Include quantified results (e.g. %, ms latency, $ saved) in experience bullets")

    return {
        "score": min(score, 100),
        "target_role": target_role,
        "feedback": feedback if feedback else ["Resume meets high ATS formatting standards!"]
    }
