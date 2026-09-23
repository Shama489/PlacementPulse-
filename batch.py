# pages/batch.py
import streamlit as st
import requests
import pandas as pd
import io

API = "http://localhost:5000"


def show():
    st.title("📁 Batch Placement Prediction")
    st.markdown(
        """
        Upload a **CSV file** containing multiple student records.  
        The system will predict placement status for each student and return a downloadable results file.
        """
    )

    # ── Template download ─────────────────────────────────────────────────────
    st.subheader("📥 Download Template")
    template_cols = [
        "age", "gender", "cgpa", "branch", "college_tier",
        "internships_count", "projects_count", "certifications_count",
        "coding_skill_score", "aptitude_score", "communication_skill_score",
        "logical_reasoning_score", "hackathons_participated", "github_repos",
        "linkedin_connections", "mock_interview_score", "attendance_percentage",
        "backlogs", "extracurricular_score", "leadership_score",
        "volunteer_experience", "sleep_hours", "study_hours_per_day",
    ]
    sample_rows = [
        [21, "Male",   7.5, "CSE",        "Tier 2", 2, 3, 2, 75.0, 70.0, 68.0, 72.0, 1, 4, 300, 70.0, 85.0, 0, 55.0, 60.0, "No",  7.0, 4.0],
        [22, "Female", 8.2, "IT",         "Tier 1", 3, 5, 4, 85.0, 80.0, 90.0, 78.0, 2, 6, 500, 82.0, 90.0, 0, 70.0, 75.0, "Yes", 8.0, 5.0],
        [23, "Male",   6.1, "Mechanical", "Tier 3", 0, 1, 1, 45.0, 50.0, 48.0, 52.0, 0, 1, 80,  45.0, 65.0, 3, 30.0, 35.0, "No",  6.0, 2.0],
    ]
    template_df = pd.DataFrame(sample_rows, columns=template_cols)
    csv_bytes = template_df.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download CSV Template",
        data=csv_bytes,
        file_name="student_template.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Upload & Predict ──────────────────────────────────────────────────────
    st.subheader("📤 Upload Student Data")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        df_input = pd.read_csv(uploaded)
        st.markdown(f"**Loaded {len(df_input):,} student records**")

        with st.expander("🔎 Preview uploaded data (first 10 rows)"):
            st.dataframe(df_input.head(10), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            with st.spinner("Predicting … this may take a moment for large files."):
                try:
                    # Re-read as bytes for multipart upload
                    uploaded.seek(0)
                    files = {"file": ("upload.csv", uploaded, "text/csv")}
                    resp = requests.post(f"{API}/predict-batch", files=files, timeout=60)

                    if resp.status_code != 200:
                        st.error(f"API Error: {resp.json().get('error', 'Unknown error')}")
                        return

                    # Parse returned CSV
                    result_df = pd.read_csv(io.StringIO(resp.content.decode()))
                    st.success(f"✅ Predictions complete for {len(result_df):,} students!")

                    # Summary metrics
                    placed_n    = (result_df["Predicted_Status"] == "Placed").sum()
                    not_placed_n = len(result_df) - placed_n
                    avg_prob    = result_df["Placement_Probability"].mean()

                    col1, col2, col3 = st.columns(3)
                    col1.metric("✅ Predicted Placed",     f"{placed_n:,}")
                    col2.metric("❌ Predicted Not Placed", f"{not_placed_n:,}")
                    col3.metric("📊 Avg Probability",      f"{avg_prob*100:.1f}%")

                    # Probability distribution chart
                    import plotly.express as px
                    fig = px.histogram(
                        result_df, x="Placement_Probability", nbins=40,
                        color="Predicted_Status",
                        color_discrete_map={"Placed": "#22c55e", "Not Placed": "#ef4444"},
                        title="Distribution of Placement Probabilities",
                    )
                    fig.update_layout(xaxis_title="Placement Probability",
                                      yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)

                    # Results table
                    st.subheader("📋 Results Preview")
                    display_cols = ["Predicted_Status", "Placement_Probability"] + \
                                   [c for c in result_df.columns
                                    if c not in ("Predicted_Status", "Placement_Probability")]
                    st.dataframe(result_df[display_cols].head(50), use_container_width=True)

                    # Download button
                    st.download_button(
                        label="⬇️ Download Full Results CSV",
                        data=resp.content,
                        file_name="placement_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(
                        f"⚠️ Could not reach the API ({e}).  \n"
                        "Make sure Flask backend is running:  \n"
                        "`python placement_predictor/backend/app.py`"
                    )

    st.markdown("---")
    st.subheader("📌 Required CSV Columns")
    cols_info = {
        "age": "int (18–30)",
        "gender": "Male / Female",
        "cgpa": "float (0.0–10.0)",
        "branch": "CSE / IT / ECE / EEE / Mechanical / Civil …",
        "college_tier": "Tier 1 / Tier 2 / Tier 3",
        "internships_count": "int",
        "projects_count": "int",
        "certifications_count": "int",
        "coding_skill_score": "float (0–100)",
        "aptitude_score": "float (0–100)",
        "communication_skill_score": "float (0–100)",
        "logical_reasoning_score": "float (0–100)",
        "hackathons_participated": "int",
        "github_repos": "int",
        "linkedin_connections": "int",
        "mock_interview_score": "float (0–100)",
        "attendance_percentage": "float (0–100)",
        "backlogs": "int",
        "extracurricular_score": "float (0–100)",
        "leadership_score": "float (0–100)",
        "volunteer_experience": "Yes / No",
        "sleep_hours": "float",
        "study_hours_per_day": "float",
    }
    cols_df = pd.DataFrame(list(cols_info.items()), columns=["Column", "Expected Format"])
    st.dataframe(cols_df, use_container_width=True, hide_index=True)
