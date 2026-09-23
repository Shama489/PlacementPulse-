# pages/predict.py
import streamlit as st
import requests

API = "http://localhost:5000"

BRANCHES = ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil", "Chemical", "Biotechnology"]
TIERS    = ["Tier 1", "Tier 2", "Tier 3"]


def show():
    st.title("🔮 Single Student Placement Prediction")
    st.markdown("Fill in the student profile below to predict placement outcome.")

    with st.form("prediction_form"):
        st.subheader("👤 Personal & Academic Details")
        c1, c2, c3 = st.columns(3)
        age    = c1.number_input("Age",           min_value=18, max_value=30, value=21)
        gender = c2.selectbox("Gender",           ["Male", "Female"])
        cgpa   = c3.number_input("CGPA",          min_value=0.0, max_value=10.0, value=7.5, step=0.1)

        c4, c5, c6 = st.columns(3)
        branch       = c4.selectbox("Branch",        BRANCHES)
        college_tier = c5.selectbox("College Tier",   TIERS)
        backlogs     = c6.number_input("Backlogs",    min_value=0, max_value=20, value=0)

        st.subheader("💼 Experience & Activities")
        c7, c8, c9, c10 = st.columns(4)
        internships    = c7.number_input("Internships",      min_value=0, max_value=10, value=1)
        projects       = c8.number_input("Projects",         min_value=0, max_value=20, value=2)
        certifications = c9.number_input("Certifications",   min_value=0, max_value=20, value=2)
        hackathons     = c10.number_input("Hackathons",      min_value=0, max_value=20, value=1)

        c11, c12, c13 = st.columns(3)
        github_repos        = c11.number_input("GitHub Repos",          min_value=0, max_value=100, value=3)
        linkedin_connections = c12.number_input("LinkedIn Connections", min_value=0, max_value=5000, value=200)
        volunteer_exp       = c13.selectbox("Volunteer Experience",     ["Yes", "No"])

        st.subheader("📊 Skill Scores (0 – 100)")
        c14, c15 = st.columns(2)
        coding_skill   = c14.slider("Coding Skill Score",         0.0, 100.0, 65.0, 0.5)
        aptitude       = c15.slider("Aptitude Score",             0.0, 100.0, 65.0, 0.5)
        communication  = c14.slider("Communication Skill Score",  0.0, 100.0, 65.0, 0.5)
        logical        = c15.slider("Logical Reasoning Score",    0.0, 100.0, 65.0, 0.5)
        mock_interview = c14.slider("Mock Interview Score",       0.0, 100.0, 65.0, 0.5)

        st.subheader("🏫 Academic Habits")
        c16, c17, c18 = st.columns(3)
        attendance   = c16.slider("Attendance %",        50.0, 100.0, 80.0, 0.5)
        study_hours  = c17.slider("Study Hours/Day",     0.0, 12.0, 4.0, 0.5)
        sleep_hours  = c18.slider("Sleep Hours/Day",     4.0, 12.0, 7.0, 0.5)

        st.subheader("🌟 Soft Skills")
        c19, c20 = st.columns(2)
        leadership      = c19.slider("Leadership Score",      0.0, 100.0, 50.0, 0.5)
        extracurricular = c20.slider("Extracurricular Score", 0.0, 100.0, 50.0, 0.5)

        submitted = st.form_submit_button("🚀 Predict Placement", use_container_width=True)

    if submitted:
        payload = {
            "age":                        age,
            "gender":                     gender,
            "cgpa":                       cgpa,
            "branch":                     branch,
            "college_tier":               college_tier,
            "internships_count":          internships,
            "projects_count":             projects,
            "certifications_count":       certifications,
            "coding_skill_score":         coding_skill,
            "aptitude_score":             aptitude,
            "communication_skill_score":  communication,
            "logical_reasoning_score":    logical,
            "hackathons_participated":    hackathons,
            "github_repos":               github_repos,
            "linkedin_connections":       linkedin_connections,
            "mock_interview_score":       mock_interview,
            "attendance_percentage":      attendance,
            "backlogs":                   backlogs,
            "extracurricular_score":      extracurricular,
            "leadership_score":           leadership,
            "volunteer_experience":       volunteer_exp,
            "sleep_hours":                sleep_hours,
            "study_hours_per_day":        study_hours,
        }

        try:
            resp = requests.post(f"{API}/predict", json=payload, timeout=10)
            result = resp.json()

            if "error" in result:
                st.error(f"API Error: {result['error']}")
                return

            prob  = result["probability"]
            pred  = result["prediction"]
            conf  = result["confidence"]

            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            if pred == "Placed":
                st.success(f"## ✅ {pred}", icon="🎉")
            else:
                st.error(f"## ❌ {pred}", icon="⚠️")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Placement Probability", f"{prob*100:.1f}%")
            col_b.metric("Confidence",            f"{conf*100:.1f}%")
            col_c.metric("Model Used",             "Best Ensemble")

            # ── Probability gauge ──────────────────────────────────────────
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                title={"text": "Placement Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#22c55e" if pred == "Placed" else "#ef4444"},
                    "steps": [
                        {"range": [0,  40], "color": "#fee2e2"},
                        {"range": [40, 60], "color": "#fef9c3"},
                        {"range": [60, 100], "color": "#dcfce7"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            # ── Profile Summary ────────────────────────────────────────────
            st.subheader("📋 Profile Summary")
            summary = {
                "CGPA": cgpa,
                "Internships": internships,
                "Projects": projects,
                "Certifications": certifications,
                "Coding Skill": f"{coding_skill:.1f}",
                "Aptitude":     f"{aptitude:.1f}",
                "Communication": f"{communication:.1f}",
                "Mock Interview": f"{mock_interview:.1f}",
                "Hackathons":    hackathons,
                "Backlogs":      backlogs,
                "Attendance":    f"{attendance:.1f}%",
            }
            import pandas as pd
            summary_df = pd.DataFrame(list(summary.items()), columns=["Attribute", "Value"])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(
                f"⚠️ Could not reach the API ({e}).  \n"
                "Make sure Flask backend is running:  \n"
                "`python placement_predictor/backend/app.py`"
            )
