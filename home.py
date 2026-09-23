# pages/home.py
import streamlit as st
import requests
import pandas as pd

API = "http://localhost:5000"


def show():
    st.title("🎓 Student Placement Prediction System")
    st.markdown(
        """
        Welcome to the **Student Placement Prediction** platform — powered by Machine Learning.
        This system analyses student academic and skill profiles to predict placement outcomes.
        """
    )

    # ── Key Metrics via API ───────────────────────────────────────────────────
    try:
        resp = requests.get(f"{API}/stats", timeout=10)
        stats = resp.json()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📚 Total Students",   f"{stats['total_students']:,}")
        col2.metric("✅ Placement Rate",    f"{stats['placed_percentage']}%")
        col3.metric("💰 Avg Salary (LPA)",  f"₹{stats['avg_salary_lpa']}")
        best = stats["best_model"]
        acc  = stats["model_results"][best]["accuracy"]
        col4.metric("🤖 Best Model Acc.",   f"{acc*100:.1f}%")

        st.markdown("---")

        # Branch placement table
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📌 Branch-wise Placement Rate")
            branch_df = pd.DataFrame.from_dict(
                stats["branch_placement_rate"], orient="index", columns=["Placement Rate (%)"]
            ).sort_values("Placement Rate (%)", ascending=False)
            st.dataframe(branch_df, use_container_width=True)

        with col_b:
            st.subheader("🏛️ College Tier Placement Rate")
            tier_df = pd.DataFrame.from_dict(
                stats["tier_placement_rate"], orient="index", columns=["Placement Rate (%)"]
            ).sort_values("Placement Rate (%)", ascending=False)
            st.dataframe(tier_df, use_container_width=True)

    except Exception as e:
        st.warning(
            f"⚠️ API not reachable ({e}). Start the Flask backend first:  \n"
            "`python placement_predictor/backend/app.py`"
        )

    st.markdown("---")
    st.subheader("🗂️ Project Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📊 EDA Dashboard**\nExplore the dataset with interactive visualisations — distributions, correlations, and placement trends.")
    with col2:
        st.success("**🔮 Single Prediction**\nFill in a student profile form and get an instant placement prediction with probability score.")
    with col3:
        st.warning("**📁 Batch Prediction**\nUpload a CSV file of multiple students and download predictions in bulk.")

    st.markdown("---")
    st.subheader("📐 Features Used for Prediction")
    features = [
        ("CGPA", "Academic performance score"),
        ("Internships", "Number of internships completed"),
        ("Projects", "Number of projects built"),
        ("Coding Skill Score", "Programming ability (0–100)"),
        ("Aptitude Score", "Logical & numerical aptitude (0–100)"),
        ("Communication Score", "Verbal & written communication (0–100)"),
        ("Logical Reasoning", "Reasoning ability (0–100)"),
        ("Mock Interview Score", "Practice interview performance"),
        ("Hackathons", "Hackathons participated in"),
        ("GitHub Repos", "Number of public repos"),
        ("LinkedIn Connections", "Professional network size"),
        ("Certifications", "Number of certifications earned"),
        ("Attendance %", "Lecture attendance percentage"),
        ("Backlogs", "Number of academic backlogs"),
    ]
    feat_df = pd.DataFrame(features, columns=["Feature", "Description"])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
