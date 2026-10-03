document.addEventListener('DOMContentLoaded', () => {
    // Global state
    let currentAnalysisData = null;
    let selectedRoleRec = null;
    let courseOutlineState = null;
    let chatHistory = [];

    // Form inputs and sliders setup
    setupSliders();
    setupResumeUploader();
    setupFormSubmission();
    setupTabs();

    function setupSliders() {
        const sliders = [
            { id: 'cgpa_input', valId: 'cgpa_val' },
            { id: 'coding_score', valId: 'coding_val' },
            { id: 'aptitude_score', valId: 'aptitude_val' },
            { id: 'communication_score', valId: 'comm_val' }
        ];

        sliders.forEach(s => {
            const input = document.getElementById(s.id);
            const valSpan = document.getElementById(s.valId);
            if (input && valSpan) {
                input.addEventListener('input', () => {
                    valSpan.textContent = input.value;
                });
            }
        });
    }

    function setupResumeUploader() {
        const fileInput = document.getElementById('resume_file_input');
        const skillsTextarea = document.getElementById('skills_input');
        const parseStatus = document.getElementById('resume_parse_status');

        if (!fileInput) return;

        fileInput.addEventListener('change', async () => {
            const file = fileInput.files[0];
            if (!file) return;

            parseStatus.style.display = 'flex';
            parseStatus.innerHTML = '<div class="spinner"></div> Parsing resume and extracting skills...';

            const formData = new FormData();
            formData.append('resume', file);

            try {
                const response = await fetch('/api/parse-resume/', {
                    method: 'POST',
                    body: formData
                });
                const res = await response.json();

                if (res.status === 'success') {
                    if (res.skills_str) {
                        skillsTextarea.value = res.skills_str;
                        parseStatus.className = 'status-item status-active';
                        parseStatus.innerHTML = `✅ Extracted ${res.skills.length} technical skills!`;
                    } else {
                        parseStatus.className = 'status-item status-offline';
                        parseStatus.innerHTML = '⚠️ No skills detected. You can enter skills manually.';
                    }
                    document.getElementById('resume_text_hidden').value = res.resume_text || '';
                } else {
                    parseStatus.className = 'status-item status-offline';
                    parseStatus.innerHTML = `❌ Error: ${res.message}`;
                }
            } catch (err) {
                parseStatus.className = 'status-item status-offline';
                parseStatus.innerHTML = `❌ Upload failed: ${err.message}`;
            }
        });
    }

    function setupFormSubmission() {
        const form = document.getElementById('career_form');
        const resultsContainer = document.getElementById('results_container');
        const emptyState = document.getElementById('empty_state');
        const analyzeBtn = document.getElementById('analyze_btn');

        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const skillsInput = document.getElementById('skills_input').value.trim();
            if (!skillsInput) {
                alert('Please enter technical skills before analyzing.');
                return;
            }

            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<div class="spinner"></div> Analyzing & Predicting...';

            const payload = {
                branch: document.getElementById('branch_select').value,
                cgpa: parseFloat(document.getElementById('cgpa_input').value),
                college_tier: parseInt(document.getElementById('college_tier_select').value),
                coding_score: parseFloat(document.getElementById('coding_score').value),
                aptitude_score: parseFloat(document.getElementById('aptitude_score').value),
                communication_score: parseFloat(document.getElementById('communication_score').value),
                internships: parseInt(document.getElementById('internships_input').value),
                projects: parseInt(document.getElementById('projects_input').value),
                backlogs: parseInt(document.getElementById('backlogs_input').value),
                dsa_skill: document.getElementById('dsa_skill_checkbox').checked,
                skills_input: skillsInput,
                resume_text: document.getElementById('resume_text_hidden').value
            };

            try {
                const response = await fetch('/api/analyze-career/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const res = await response.json();

                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '🚀 Analyze & Predict Career';

                if (res.status === 'success') {
                    currentAnalysisData = res;
                    emptyState.style.display = 'none';
                    resultsContainer.style.display = 'block';
                    renderResults(res);
                } else {
                    alert(`Analysis failed: ${res.message}`);
                }
            } catch (err) {
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '🚀 Analyze & Predict Career';
                alert(`Request failed: ${err.message}`);
            }
        });
    }

    function renderResults(data) {
        renderDualRecommendations(data.dual_insights);
        renderTopRecommendations(data.skill_recs);
        renderRankingsBarChart(data.skill_recs);
        setupRoleDropdown(data.skill_recs);
    }

    function renderDualRecommendations(dual) {
        const container = document.getElementById('dual_recommendations_container');
        const content = document.getElementById('dual_insights_content');
        if (!container || !content || !dual) return;

        container.style.display = 'block';

        const isDisc = dual.is_discrepancy;
        const savedCache = dual.discrepancy_saved_to_cache;
        const learnedApplied = dual.learned_pattern_applied;

        let statusBadge = '';
        if (isDisc) {
            statusBadge = `<span class="badge badge-normal" style="background: rgba(239, 68, 68, 0.2); color: #f87171; font-weight: 700;">⚠️ ML vs AI Recommendation Discrepancy Detected</span>`;
        } else {
            statusBadge = `<span class="badge badge-normal" style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-weight: 700;">✅ ML & AI Recommendation Match</span>`;
        }

        let cacheBadge = '';
        if (savedCache) {
            cacheBadge = `<span class="badge badge-normal" style="background: rgba(99, 102, 241, 0.25); color: #a5b4fc; font-weight: 700;">💾 Saved Discrepancy to Cache Memory</span>`;
        }
        if (learnedApplied) {
            cacheBadge += `<span class="badge badge-normal" style="background: rgba(245, 158, 11, 0.25); color: #fbbf24; font-weight: 700;">💡 Applied Learned Pattern from Cache Memory</span>`;
        }

        let html = `
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; align-items: center;">
                ${statusBadge}
                ${cacheBadge}
                <span style="font-size: 0.8rem; color: #9ca3af; margin-left: auto;">Total Cache Memory Cases: <strong>${dual.total_cached_discrepancies || 0}</strong></span>
            </div>

            <div class="form-row-2" style="margin-bottom: 12px; gap: 14px;">
                <div style="background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 10px; border: 1px solid rgba(52, 211, 153, 0.3);">
                    <div style="font-size: 0.8rem; color: #34d399; font-weight: 700; text-transform: uppercase;">📊 ML Model Recommendation</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">${dual.ml_top_role || 'N/A'}</div>
                    <div style="font-size: 0.82rem; color: #9ca3af;">Predicted via Scikit-Learn Profile Classifier & TF-IDF Skill Model</div>
                </div>

                <div style="background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 10px; border: 1px solid rgba(99, 102, 241, 0.4);">
                    <div style="font-size: 0.8rem; color: #818cf8; font-weight: 700; text-transform: uppercase;">⚡ Groq AI Cloud Recommendation</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">${dual.ai_top_role || 'N/A'}</div>
                    <div style="font-size: 0.85rem; color: #d1d5db;">${dual.ai_reasoning || ''}</div>
                </div>
            </div>
        `;

        if (dual.cached_insight_msg) {
            html += `
                <div style="background: rgba(245, 158, 11, 0.08); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(245, 158, 11, 0.3); font-size: 0.85rem; color: #fbbf24;">
                    🧠 <strong>Cache Memory Insights:</strong> ${dual.cached_insight_msg}
                </div>
            `;
        }

        content.innerHTML = html;
    }


    function renderTopRecommendations(skillRecs) {
        const container = document.getElementById('top_recs_container');
        if (!container || !skillRecs || skillRecs.length === 0) return;

        const top3 = skillRecs.slice(0, 3);
        let html = '';

        top3.forEach((rec, idx) => {
            const parentHtml = rec.parent_role ? `<div class="rec-sub-role">Sub-role of ${rec.parent_role}</div>` : '';
            html += `
                <div class="rec-card">
                    <span class="rec-rank-tag">🏆 #${idx + 1} Recommendation</span>
                    <h3 class="rec-title">${rec.role}</h3>
                    ${parentHtml}
                    <p class="rec-fit-score">Combined Fit: <strong>${rec.score}% Match</strong></p>
                    <div class="rec-breakdown">
                        Academic Fit: ${rec.profile_score}%<br/>
                        Skill Match: ${rec.skill_score}%<br/>
                        Model Conf: ${rec.skills_ml_score}%
                    </div>
                    <button class="btn-secondary" onclick="window.fetchWhyWhat('${rec.role}', '${(currentAnalysisData.combined_skills || []).join(',')}', 'why_what_box_${idx}')">
                        🌐 Why & What Insights
                    </button>
                    <div id="why_what_box_${idx}" style="margin-top: 10px; font-size: 0.8rem; display: none;"></div>
                </div>
            `;
        });

        container.innerHTML = html;
    }

    window.fetchWhyWhat = async (role, skillsStr, targetId) => {
        const box = document.getElementById(targetId);
        if (!box) return;

        box.style.display = 'block';
        box.innerHTML = '<div class="spinner"></div> Fetching live insights...';

        try {
            const res = await fetch(`/api/why-what/?role=${encodeURIComponent(role)}&skills=${encodeURIComponent(skillsStr)}`);
            const data = await res.json();
            if (data.status === 'success') {
                const ins = data.insights;
                box.innerHTML = `
                    <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                        <strong style="color: #60a5fa;">❓ What it is:</strong> ${ins.what || 'N/A'}<br/><br/>
                        <strong style="color: #34d399;">🎯 Why suggest:</strong> ${ins.why || 'N/A'}
                    </div>
                `;
            } else {
                box.innerHTML = `<span style="color: #ef4444;">Failed to fetch insights.</span>`;
            }
        } catch (err) {
            box.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
        }
    };

    function renderRankingsBarChart(skillRecs) {
        const container = document.getElementById('rankings_bar_container');
        if (!container || !skillRecs) return;

        let html = '';
        skillRecs.slice(0, 10).forEach((rec, idx) => {
            const isTop = idx === 0;
            const barClass = isTop ? 'rec-bar-fill rec-bar-fill-top' : 'rec-bar-fill';
            const subSuffix = rec.parent_role ? `<span style="color: #a5b4fc; font-size: 0.8rem;"> (Sub-role of ${rec.parent_role})</span>` : '';

            html += `
                <div class="rec-item">
                    <div class="rec-label-container">
                        <span>${rec.role}${subSuffix}</span>
                        <span style="color: #9ca3af; font-size: 0.8rem;">(Acad Fit: ${rec.profile_score}% | Skill: ${rec.skill_score}%)</span>
                        <span>${rec.score}% Match</span>
                    </div>
                    <div class="rec-bar-bg">
                        <div class="${barClass}" style="width: ${rec.score}%;"></div>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    }

    function setupRoleDropdown(skillRecs) {
        const select = document.getElementById('role_detail_select');
        if (!select) return;

        select.innerHTML = '';
        skillRecs.forEach(rec => {
            const opt = document.createElement('option');
            opt.value = rec.role;
            opt.textContent = rec.parent_role ? `${rec.role} (Sub-role of ${rec.parent_role})` : rec.role;
            select.appendChild(opt);
        });

        select.addEventListener('change', () => {
            const selectedRole = select.value;
            selectedRoleRec = skillRecs.find(r => r.role === selectedRole);
            renderSelectedRoleDetails(selectedRoleRec);
        });

        if (skillRecs.length > 0) {
            selectedRoleRec = skillRecs[0];
            renderSelectedRoleDetails(selectedRoleRec);
        }
    }

    function renderSelectedRoleDetails(rec) {
        if (!rec) return;

        // Render Gauge
        const score = rec.skill_score;
        const offset = 263.89 * (1 - score / 100);
        const circle = document.getElementById('gauge_circle');
        const scoreText = document.getElementById('gauge_score_text');
        const badge = document.getElementById('gauge_badge');

        if (circle) circle.setAttribute('stroke-dashoffset', offset);
        if (scoreText) scoreText.textContent = `${Math.round(score)}%`;

        if (badge) {
            if (score >= 75) {
                badge.className = 'score-badge score-high';
                badge.textContent = 'Excellent Match';
                if (circle) circle.setAttribute('stroke', '#10b981');
            } else if (score >= 50) {
                badge.className = 'score-badge score-mid';
                badge.textContent = 'Good Match';
                if (circle) circle.setAttribute('stroke', '#f59e0b');
            } else {
                badge.className = 'score-badge score-low';
                badge.textContent = 'Needs Improvement';
                if (circle) circle.setAttribute('stroke', '#ef4444');
            }
        }

        // Render Tabs
        renderSkillsGapTab(rec);
        renderRequiredSkillsTab(rec);
        renderYourSkillsTab();
        setupCourseTab(rec);

        // Auto-fetch role-specific live insights & jobs for the selected role
        window.fetchMarketInsights();
        window.fetchLiveJobs();
    }

    function renderSkillsGapTab(rec) {
        const matchedContainer = document.getElementById('matched_skills_badges');
        const missingContainer = document.getElementById('missing_skills_badges');

        if (matchedContainer) {
            if (rec.matched_skills && rec.matched_skills.length > 0) {
                matchedContainer.innerHTML = rec.matched_skills.map(s => `<span class="badge badge-matched">${s}</span>`).join('');
            } else {
                matchedContainer.innerHTML = '<span style="color: #9ca3af; font-size: 0.85rem;">No matched skills found.</span>';
            }
        }

        if (missingContainer) {
            if (rec.missing_skills && rec.missing_skills.length > 0) {
                missingContainer.innerHTML = rec.missing_skills.map(s => `<span class="badge badge-missing">${s}</span>`).join('');
            } else {
                missingContainer.innerHTML = '<span style="color: #34d399; font-size: 0.85rem;">🎉 You match all required skills for this role!</span>';
            }
        }
    }

    function renderRequiredSkillsTab(rec) {
        const container = document.getElementById('required_skills_badges');
        if (container) {
            if (rec.required_skills && rec.required_skills.length > 0) {
                container.innerHTML = rec.required_skills.map(s => `<span class="badge badge-normal">${s}</span>`).join('');
            } else {
                container.innerHTML = '<span style="color: #9ca3af; font-size: 0.85rem;">No skill breakdown cached.</span>';
            }
        }
    }

    function renderYourSkillsTab() {
        const container = document.getElementById('your_skills_badges');
        if (container && currentAnalysisData) {
            const skills = currentAnalysisData.combined_skills || [];
            container.innerHTML = skills.map(s => `<span class="badge badge-normal">${s}</span>`).join('');
        }
    }

    function resetMarketInsightsTab() {
        const container = document.getElementById('market_insights_container');
        if (container) container.innerHTML = '<button class="btn-secondary" onclick="window.fetchMarketInsights()">📈 Load Live Market Data (Tavily)</button>';
    }

    window.fetchMarketInsights = async () => {
        if (!selectedRoleRec) return;
        const container = document.getElementById('market_insights_container');
        container.innerHTML = '<div class="spinner-overlay"><div class="spinner"></div> Fetching distinct market insights via Tavily API...</div>';

        try {
            const res = await fetch(`/api/market-insights/?role=${encodeURIComponent(selectedRoleRec.role)}`);
            const data = await res.json();
            if (data.status === 'success') {
                const ins = data.insights;
                container.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                        <div class="glass-card" style="padding: 16px; margin: 0;">
                            <h5 style="color: #60a5fa; margin-bottom: 6px;">💰 Avg Compensation for ${selectedRoleRec.role}</h5>
                            <p style="font-size: 1.1rem; font-weight: 700;">${ins.average_salary || 'N/A'}</p>
                        </div>
                        <div class="glass-card" style="padding: 16px; margin: 0;">
                            <h5 style="color: #818cf8; margin-bottom: 6px;">🚀 Hiring Trends</h5>
                            <p style="font-size: 0.85rem; line-height: 1.4;">${ins.market_trends || 'N/A'}</p>
                        </div>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <strong style="color: #a5b4fc;">🏢 Active Hiring Companies for ${selectedRoleRec.role}:</strong>
                        <ul style="margin-left: 20px; font-size: 0.85rem; margin-top: 4px;">
                            ${(ins.top_companies || []).map(c => `<li>${c}</li>`).join('') || '<li>N/A</li>'}
                        </ul>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <strong style="color: #a5b4fc;">🎓 Recommended Certifications:</strong>
                        <ul style="margin-left: 20px; font-size: 0.85rem; margin-top: 4px;">
                            ${(ins.certifications || []).map(c => `<li>${c}</li>`).join('') || '<li>N/A</li>'}
                        </ul>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <strong style="color: #34d399;">💡 Practical Project Ideas:</strong>
                        <ul style="margin-left: 20px; font-size: 0.85rem; margin-top: 4px;">
                            ${(ins.project_ideas || []).map(p => `<li>${p}</li>`).join('') || '<li>N/A</li>'}
                        </ul>
                    </div>
                    <div>
                        <strong style="color: #f87171;">🗣️ Key Interview Prep Topics:</strong>
                        <ul style="margin-left: 20px; font-size: 0.85rem; margin-top: 4px;">
                            ${(ins.interview_tips || []).map(t => `<li>${t}</li>`).join('') || '<li>N/A</li>'}
                        </ul>
                    </div>
                `;
            } else {
                container.innerHTML = `<span style="color: #ef4444;">Failed: ${data.message}</span>`;
            }
        } catch (err) {
            container.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
        }
    };

    function resetLiveJobsTab() {
        const container = document.getElementById('live_jobs_container');
        if (container) container.innerHTML = '<button class="btn-secondary" onclick="window.fetchLiveJobs()">💼 Search Live Openings (Tavily)</button>';
    }

    window.fetchLiveJobs = async () => {
        if (!selectedRoleRec) return;
        const container = document.getElementById('live_jobs_container');
        container.innerHTML = '<div class="spinner-overlay"><div class="spinner"></div> Searching live job openings for ' + selectedRoleRec.role + ' via Tavily...</div>';

        try {
            const res = await fetch(`/api/live-jobs/?role=${encodeURIComponent(selectedRoleRec.role)}`);
            const data = await res.json();
            if (data.status === 'success') {
                const listings = data.data.listings || [];
                if (listings.length === 0) {
                    container.innerHTML = '<p style="color: #9ca3af;">No active jobs found currently.</p>';
                    return;
                }
                let html = `<p style="margin-bottom: 12px; font-size: 0.85rem;">Found <strong>${listings.length}</strong> active openings for <strong>${selectedRoleRec.role}</strong>:</p>`;
                listings.forEach(job => {
                    const isIntern = job.type === 'Internship';
                    const borderCol = isIntern ? '#a855f7' : '#10b981';
                    const typeBg = isIntern ? 'rgba(168, 85, 247, 0.15)' : 'rgba(16, 185, 129, 0.15)';
                    const typeCol = isIntern ? '#c084fc' : '#6ee7b7';

                    html += `
                        <div class="glass-card" style="padding: 14px; margin-bottom: 10px; border-left: 4px solid ${borderCol};">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <h5 style="color: ${typeCol}; font-size: 1rem; margin-bottom: 4px;">${job.title}</h5>
                                <span style="background: ${typeBg}; color: ${typeCol}; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">${job.type || 'Job'}</span>
                            </div>
                            <p style="font-size: 0.85rem; color: #e5e7eb; margin-bottom: 6px;">🏢 <strong>${job.company}</strong> | ${job.platform}</p>
                            <p style="font-size: 0.8rem; color: #9ca3af; margin-bottom: 8px;">${job.description || ''}</p>
                            <a href="${job.url}" target="_blank" class="btn-secondary" style="display: inline-block; font-size: 0.75rem; text-decoration: none;">🔗 Apply on ${job.platform}</a>
                        </div>
                    `;
                });
                container.innerHTML = html;
            } else {
                container.innerHTML = `<span style="color: #ef4444;">Failed: ${data.message}</span>`;
            }
        } catch (err) {
            container.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
        }
    };

    function setupCourseTab(rec) {
        const goalSelect = document.getElementById('course_goal_select');
        if (!goalSelect || !rec) return;

        goalSelect.innerHTML = '';
        const optRole = document.createElement('option');
        optRole.value = rec.role;
        optRole.textContent = `Learn ${rec.role}`;
        goalSelect.appendChild(optRole);

        (rec.missing_skills || []).forEach(ms => {
            const opt = document.createElement('option');
            opt.value = ms;
            opt.textContent = `Master ${ms}`;
            goalSelect.appendChild(opt);
        });

        checkOllamaStatus();
    }

    async function checkOllamaStatus() {
        const statusBox = document.getElementById('ollama_status_box');
        if (!statusBox) return;

        try {
            const res = await fetch('/api/ollama-status/');
            const data = await res.json();
            if (data.connected) {
                statusBox.className = 'status-item status-active';
                statusBox.innerHTML = `✅ Local Ollama AI Connected (${(data.models || []).length} models)`;
                setupOllamaModelSelect(data.models || []);
            } else {
                statusBox.className = 'status-item status-offline';
                statusBox.innerHTML = 'ℹ️ Local Ollama Offline (Start `ollama serve` for local chat)';
            }
        } catch (err) {
            statusBox.className = 'status-item status-offline';
            statusBox.innerHTML = 'ℹ️ Ollama status check failed';
        }
    }

    async function checkOllamaModels() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            const select = document.getElementById('course_model_select');
            if (select && data.models && data.models.length > 0) {
                // Keep Groq as top option, add all detected local Ollama models
                select.innerHTML = '<option value="groq" selected>⚡ Groq AI Cloud (Llama 3.3 70B)</option>';
                data.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = `ollama:${m.name}`;
                    opt.textContent = `🦙 Ollama Local (${m.name})`;
                    select.appendChild(opt);
                });
            }
        } catch (err) {
            console.log("Local Ollama check skipped:", err);
        }
    }
    checkOllamaModels();

    function setupOllamaModelSelect(models) {
        const select = document.getElementById('ollama_model_select');
        if (!select) return;

        select.innerHTML = '';
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = m.name;
            select.appendChild(opt);
        });
    }

    // Setup Tabs switcher
    function setupTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.getAttribute('data-tab');
                tabBtns.forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));

                btn.classList.add('active');
                const content = document.getElementById(targetTab);
                if (content) content.classList.add('active');
            });
        });
    }

    // Course Outline Generation Handler
    const genCourseBtn = document.getElementById('generate_course_btn');
    if (genCourseBtn) {
        genCourseBtn.addEventListener('click', async () => {
            const goalSelect = document.getElementById('course_goal_select');
            const modelSelect = document.getElementById('course_model_select');
            const diffSelect = document.getElementById('course_difficulty_select');
            const durSelect = document.getElementById('course_duration_select');
            const styleSelect = document.getElementById('course_style_select');

            const goal = goalSelect ? goalSelect.value : 'Software Engineering';
            const model = modelSelect ? modelSelect.value : 'groq';
            const difficulty = diffSelect ? diffSelect.value : 'Intermediate';
            const duration_weeks = durSelect ? durSelect.value : 'auto';
            const learning_style = styleSelect ? styleSelect.value : 'Project-Based';

            const courseContainer = document.getElementById('course_roadmap_container');
            const progressWrapper = document.getElementById('course_progress_wrapper');

            if (progressWrapper) progressWrapper.style.display = 'block';
            courseContainer.innerHTML = `<div class="spinner-overlay"><div class="spinner"></div> Evaluating target subject "${goal}" & generating tailored ${duration_weeks === 'auto' ? 'Auto-Detected' : duration_weeks + '-Week'} (${difficulty}) curriculum via Groq AI Cloud...</div>`;

            try {
                const res = await fetch('/api/course-outline/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        learning_goal: goal,
                        model: model,
                        difficulty: difficulty,
                        duration_weeks: duration_weeks,
                        learning_style: learning_style
                    })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    courseOutlineState = data.outline;
                    renderCourseOutline(data.outline, goal, model);
                } else {
                    courseContainer.innerHTML = `<span style="color: #ef4444;">Failed to generate course: ${data.message}</span>`;
                }
            } catch (err) {
                courseContainer.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
            }
        });
    }

    function renderCourseOutline(outline, goal, model) {
        const container = document.getElementById('course_roadmap_container');
        if (!container) return;

        const estHours = outline.estimated_hours || ((outline.duration_weeks || 4) * 12);

        let html = `
            <div class="glass-card" style="padding: 20px; border-color: rgba(99, 102, 241, 0.4); margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span class="badge badge-normal" style="background: rgba(99, 102, 241, 0.2); color: #a5b4fc; font-weight: 700;">${outline.difficulty || 'Intermediate'} Level</span>
                        <span class="badge badge-normal" style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-weight: 700;">⏱️ ${outline.duration_weeks || 4} Weeks (${estHours} Study Hours)</span>
                        <span class="badge badge-normal" style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; font-weight: 700;">🎯 Subject: ${outline.subject_target || goal}</span>
                        <span class="badge badge-normal" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; font-weight: 700;">⚡ Groq AI (Llama 3.3 70B)</span>
                        <h3 style="color: #6ee7b7; margin: 10px 0 4px 0; font-size: 1.4rem;">${outline.title || goal}</h3>
                        <p style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 12px;">${outline.tagline || outline.description || ''}</p>
                    </div>
                </div>
        `;

        if (outline.prerequisites && outline.prerequisites.length > 0) {
            html += `<div style="margin-bottom: 10px;"><strong>📋 Prerequisites:</strong> ${outline.prerequisites.map(p => `<span class="badge badge-normal">${p}</span>`).join('')}</div>`;
        }

        if (outline.learning_outcomes && outline.learning_outcomes.length > 0) {
            html += `<div style="margin-bottom: 10px;"><strong>🎯 Learning Outcomes:</strong><ul style="margin-left: 20px; font-size: 0.85rem; color: #d1d5db;">${outline.learning_outcomes.map(o => `<li>${o}</li>`).join('')}</ul></div>`;
        }

        if (outline.capstone_project) {
            html += `<div style="background: rgba(16, 185, 129, 0.08); padding: 12px; border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.2);"><strong>🚀 Capstone Project Milestone:</strong> <span style="font-size: 0.88rem; color: #e2e8f0;">${outline.capstone_project}</span></div>`;
        }

        html += '</div><h5 style="color: #818cf8; margin-bottom: 12px;">📅 Weekly Syllabus & Daily Lessons</h5>';

        (outline.weeks || []).forEach(w => {
            html += `
                <div class="accordion-item">
                    <div class="accordion-header" onclick="this.nextElementSibling.classList.toggle('active')">
                        <span>Week ${w.week}: ${w.title} (${w.focus || 'Theory'})</span>
                        <span>▼</span>
                    </div>
                    <div class="accordion-body">
                        <p style="margin-bottom: 10px;"><strong>Core Concepts:</strong> ${(w.concepts || []).join(', ')}</p>
                        <button class="btn-secondary" onclick="window.loadWeekDetails('${goal}', ${w.week}, '${w.title.replace(/'/g, "\\'")}', 'week_content_${w.week}', '${outline.difficulty || 'Intermediate'}', '${model}', '${outline.learning_style || 'Project-Based'}')">
                            Load Daily Breakdown for Week ${w.week}
                        </button>
                        <div id="week_content_${w.week}" style="margin-top: 12px;"></div>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    }

    window.loadWeekDetails = async (goal, weekNum, weekTitle, targetId, difficulty, model, learningStyle) => {
        const box = document.getElementById(targetId);
        if (!box) return;

        const activeModel = model || (document.getElementById('course_model_select') ? document.getElementById('course_model_select').value : 'groq');
        const style = learningStyle || (document.getElementById('course_style_select') ? document.getElementById('course_style_select').value : 'Project-Based');
        box.innerHTML = '<div class="spinner"></div> Generating daily tasks via AI engine...';

        try {
            const res = await fetch('/api/week-details/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    learning_goal: goal,
                    model: activeModel,
                    week_num: weekNum,
                    week_title: weekTitle,
                    difficulty: difficulty,
                    learning_style: style
                })
            });
            const data = await res.json();
            if (data.status === 'success') {
                const days = data.week_details.days || data.days || [];
                let html = '<div style="margin-top: 8px;">';
                days.forEach(d => {
                    html += `
                        <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.06);">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span><strong>Day ${d.day}: ${d.title}</strong> (${d.duration_minutes} mins)</span>
                                <label style="font-size: 0.8rem; color: #34d399; cursor: pointer;">
                                    <input type="checkbox" onchange="window.toggleDayCompleted(this)"> Mark Done
                                </label>
                            </div>
                            <div id="day_box_${weekNum}_${d.day}">
                                <button class="btn-secondary" style="font-size: 0.75rem; margin-top: 8px;" onclick="window.loadDayDetails('${goal}', '${d.title.replace(/'/g, "\\'")}', ${d.day}, '${d.task_type}', ${d.duration_minutes}, '${difficulty}', 'day_box_${weekNum}_${d.day}', '${activeModel}', '${style}')">
                                    Load Lesson, Topic Videos & Quiz
                                </button>
                            </div>
                        </div>
                    `;
                });
                html += '</div>';
                box.innerHTML = html;
            } else {
                box.innerHTML = `<span style="color: #ef4444;">Failed to load week details.</span>`;
            }
        } catch (err) {
            box.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
        }
    };

    window.toggleDayCompleted = (chk) => {
        const checkboxes = document.querySelectorAll('input[type="checkbox"][onchange*="toggleDayCompleted"]');
        if (!checkboxes || checkboxes.length === 0) return;

        let checkedCount = 0;
        checkboxes.forEach(c => { if (c.checked) checkedCount++; });

        const pct = Math.round((checkedCount / checkboxes.length) * 100);
        const fill = document.getElementById('course_progress_fill');
        const text = document.getElementById('course_progress_text');

        if (fill) fill.style.width = pct + '%';
        if (text) text.textContent = pct + '%';
    };

    window.loadDayDetails = async (goal, dayTitle, dayNum, taskType, duration, difficulty, targetId, model, learningStyle) => {
        const box = document.getElementById(targetId);
        if (!box) return;

        const activeModel = model || (document.getElementById('course_model_select') ? document.getElementById('course_model_select').value : 'groq');
        const style = learningStyle || (document.getElementById('course_style_select') ? document.getElementById('course_style_select').value : 'Project-Based');
        box.innerHTML = '<div class="spinner"></div> Generating topic-by-topic lesson content & dedicated video resources...';

        try {
            const res = await fetch('/api/day-details/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    learning_goal: goal,
                    model: activeModel,
                    day_title: dayTitle,
                    day_num: dayNum,
                    task_type: taskType,
                    duration_minutes: duration,
                    difficulty: difficulty,
                    learning_style: style
                })
            });

            const data = await res.json();
            if (data.status === 'success') {
                const dData = data.day_details || data;
                const lab = dData.hands_on_lab || {};
                const quiz = dData.quiz || {};
                const uid = Math.random().toString(36).substring(2, 9);

                box.innerHTML = `
                    <div style="margin-top: 10px;">
                        <div style="margin-bottom: 8px;">
                            <button class="day-subtab-btn active" onclick="window.switchDaySubTab('${uid}_lesson', this)">📖 Topics & Videos</button>
                            <button class="day-subtab-btn" onclick="window.switchDaySubTab('${uid}_lab', this)">💻 Hands-on Lab</button>
                            <button class="day-subtab-btn" onclick="window.switchDaySubTab('${uid}_quiz', this)">❓ Quiz Checkpoint</button>
                        </div>

                        <!-- Subtab 1: Lesson Topic-by-Topic with Videos -->
                        <div id="${uid}_lesson" class="day-subtab-content" style="display: block;">
                            <p style="font-size: 0.88rem; color: #e2e8f0; line-height: 1.5; margin-bottom: 12px;">${dData.description || ''}</p>
                            
                            ${(dData.topics || []).map(top => `
                                <div class="topic-card" style="background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 10px; border: 1px solid rgba(99, 102, 241, 0.25); margin-bottom: 12px;">
                                    <div class="topic-card-title" style="color: #6ee7b7; font-weight: 700; font-size: 1.05rem;">📌 ${top.topic_name || 'Micro-Topic'}</div>
                                    <div class="topic-card-summary" style="color: #d1d5db; font-size: 0.88rem; margin: 6px 0;">${top.summary || ''}</div>
                                    ${top.code_snippet ? `<div class="code-block" style="margin: 8px 0;"><pre><code>${top.code_snippet}</code></pre></div>` : ''}
                                    
                                    <!-- Dedicated Video & Resource for this Topic -->
                                    <div style="margin-top: 10px; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                                        ${top.video ? `
                                            <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
                                                ${top.video.thumbnail ? `<img src="${top.video.thumbnail}" alt="Thumbnail" style="width: 120px; height: 68px; object-fit: cover; border-radius: 6px;">` : ''}
                                                <div style="flex: 1;">
                                                    <div style="font-size: 0.85rem; font-weight: 600; color: #f43f5e; margin-bottom: 4px;">🎬 Targeted Video Tutorial</div>
                                                    <a href="${top.video.url}" target="_blank" class="btn-topic-video" style="display: inline-block; background: #e11d48; color: white; padding: 6px 12px; border-radius: 6px; text-decoration: none; font-size: 0.8rem; font-weight: 600;">
                                                        ▶ Watch: ${top.video.title}
                                                    </a>
                                                </div>
                                            </div>
                                        ` : ''}
                                        ${top.doc ? `
                                            <div style="margin-top: 8px; font-size: 0.82rem;">
                                                <span style="color: #38bdf8; font-weight: 600;">📖 Documentation Reference: </span>
                                                <a href="${top.doc.url}" target="_blank" style="color: #93c5fd; text-decoration: underline;">
                                                    ${top.doc.title}
                                                </a>
                                            </div>
                                        ` : ''}
                                    </div>
                                </div>
                            `).join('')}
                        </div>

                        <!-- Subtab 2: Hands-on Lab -->
                        <div id="${uid}_lab" class="day-subtab-content" style="display: none;">
                            <div class="lab-container">
                                <div class="lab-title">💻 ${lab.title || 'Coding Challenge'}</div>
                                <p style="font-size: 0.85rem; margin-bottom: 6px;"><strong>Challenge:</strong> ${lab.challenge || lab.instructions || ''}</p>
                                <p style="font-size: 0.82rem; color: #9ca3af; margin-bottom: 6px;"><strong>Instructions:</strong> ${lab.instructions || ''}</p>
                                ${lab.hint ? `<p style="font-size: 0.8rem; color: #f59e0b;">💡 <strong>Hint:</strong> ${lab.hint}</p>` : ''}
                            </div>
                        </div>

                        <!-- Subtab 3: Quiz -->
                        <div id="${uid}_quiz" class="day-subtab-content" style="display: none;">
                            <div class="quiz-container">
                                <div class="quiz-question">❓ ${quiz.question || 'Knowledge Assessment'}</div>
                                ${(quiz.options || []).map(opt => `<div class="quiz-option">${opt}</div>`).join('')}
                                <div style="margin-top: 10px;">
                                    <button class="btn-secondary" style="font-size: 0.75rem; padding: 4px 10px;" onclick="this.nextElementSibling.style.display='block'">Show Solution & Explanation</button>
                                    <div style="display: none; margin-top: 8px; font-size: 0.82rem; color: #34d399; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 6px;">
                                        <strong>Correct Answer:</strong> ${quiz.answer || ''}<br/>
                                        <strong>Explanation:</strong> ${quiz.explanation || ''}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;

            } else {
                box.innerHTML = `<span style="color: #ef4444;">Failed: ${data.message}</span>`;
            }
        } catch (err) {
            box.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
        }
    };

    window.switchDaySubTab = (contentId, btn) => {
        const parent = btn.parentElement;
        parent.querySelectorAll('.day-subtab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const grandParent = parent.parentElement;
        grandParent.querySelectorAll('.day-subtab-content').forEach(c => c.style.display = 'none');

        const target = document.getElementById(contentId);
        if (target) target.style.display = 'block';
    };


    // Chatbot send handler
    const chatSendBtn = document.getElementById('chat_send_btn');
    const chatInput = document.getElementById('chat_input');
    const chatMessagesContainer = document.getElementById('chat_messages');

    if (chatSendBtn && chatInput) {
        chatSendBtn.addEventListener('click', sendChatMessage);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendChatMessage();
        });
    }

    async function sendChatMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        const modelSelect = document.getElementById('ollama_model_select');
        const selectedModel = modelSelect ? modelSelect.value : '';

        chatInput.value = '';
        appendChatMessage('user', text);

        if (!selectedModel) {
            appendChatMessage('assistant', '⚠️ Local Ollama is offline or no model selected.');
            return;
        }

        try {
            chatHistory.push({ role: 'user', content: text });
            const res = await fetch('/api/ollama-chat/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: selectedModel,
                    title: courseOutlineState ? courseOutlineState.title : 'Course',
                    goal: selectedRoleRec ? selectedRoleRec.role : 'Learning',
                    messages: chatHistory
                })
            });
            const data = await res.json();
            if (data.status === 'success') {
                appendChatMessage('assistant', data.reply);
                chatHistory.push({ role: 'assistant', content: data.reply });
            } else {
                appendChatMessage('assistant', `❌ Chat failed: ${data.message}`);
            }
        } catch (err) {
            appendChatMessage('assistant', `❌ Error: ${err.message}`);
        }
    }

    function appendChatMessage(role, content) {
        if (!chatMessagesContainer) return;
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-bubble chat-${role}`;
        msgDiv.textContent = content;
        chatMessagesContainer.appendChild(msgDiv);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    }
});
