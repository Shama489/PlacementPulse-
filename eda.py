# pages/eda.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "dataset.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def show():
    st.title("📊 Exploratory Data Analysis Dashboard")
    df = load_data()

    # ── Filters ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔍 Filters")
    branches = ["All"] + sorted(df["branch"].unique().tolist())
    sel_branch = st.sidebar.selectbox("Branch", branches)
    tiers = ["All"] + sorted(df["college_tier"].unique().tolist())
    sel_tier = st.sidebar.selectbox("College Tier", tiers)

    dff = df.copy()
    if sel_branch != "All":
        dff = dff[dff["branch"] == sel_branch]
    if sel_tier != "All":
        dff = dff[dff["college_tier"] == sel_tier]

    placed   = dff[dff["placement_status"] == "Placed"]
    unplaced = dff[dff["placement_status"] == "Not Placed"]

    st.markdown(f"**Showing {len(dff):,} students** | Placed: {len(placed):,} | Not Placed: {len(unplaced):,}")
    st.markdown("---")

    # ── Row 1: Placement distribution + Branch breakdown ─────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Placement Distribution")
        fig = px.pie(
            dff, names="placement_status", hole=0.4,
            color_discrete_map={"Placed": "#22c55e", "Not Placed": "#ef4444"},
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Placement by Branch")
        branch_stats = (
            dff.groupby(["branch", "placement_status"])
            .size().reset_index(name="count")
        )
        fig = px.bar(
            branch_stats, x="branch", y="count", color="placement_status",
            barmode="group",
            color_discrete_map={"Placed": "#22c55e", "Not Placed": "#ef4444"},
        )
        fig.update_layout(legend_title="Status")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: CGPA distribution + Salary distribution ───────────────────────
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("CGPA Distribution by Placement")
        fig = px.histogram(
            dff, x="cgpa", color="placement_status", nbins=40,
            barmode="overlay", opacity=0.7,
            color_discrete_map={"Placed": "#22c55e", "Not Placed": "#ef4444"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Salary Package Distribution (Placed)")
        sal_df = placed[placed["salary_package_lpa"] > 0]
        fig = px.histogram(sal_df, x="salary_package_lpa", nbins=50,
                           color_discrete_sequence=["#3b82f6"])
        fig.update_layout(xaxis_title="Salary (LPA)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Skill scores box plots ─────────────────────────────────────────
    st.subheader("Skill Scores by Placement Status")
    skill_cols = ["coding_skill_score", "aptitude_score",
                  "communication_skill_score", "logical_reasoning_score",
                  "mock_interview_score"]
    skill_df = dff[skill_cols + ["placement_status"]].melt(
        id_vars="placement_status", var_name="Skill", value_name="Score"
    )
    skill_df["Skill"] = skill_df["Skill"].str.replace("_score", "").str.replace("_", " ").str.title()
    fig = px.box(
        skill_df, x="Skill", y="Score", color="placement_status",
        color_discrete_map={"Placed": "#22c55e", "Not Placed": "#ef4444"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Correlation heatmap ─────────────────────────────────────────────
    st.subheader("Feature Correlation Heatmap")
    num_df = dff.select_dtypes(include=np.number).drop(
        columns=["student_id", "salary_package_lpa"], errors="ignore"
    )
    corr = num_df.corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 5: Gender + College Tier ──────────────────────────────────────────
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Placement by Gender")
        g_df = dff.groupby(["gender", "placement_status"]).size().reset_index(name="count")
        fig = px.bar(g_df, x="gender", y="count", color="placement_status",
                     barmode="group",
                     color_discrete_map={"Placed": "#22c55e", "Not Placed": "#ef4444"})
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.subheader("Placement by College Tier")
        t_df = dff.groupby(["college_tier", "placement_status"]).size().reset_index(name="count")
        fig = px.bar(t_df, x="college_tier", y="count", color="placement_status",
                     barmode="group",
                     color_discrete_map={"Placed": "#22c55e", "Not Placed": "#ef4444"})
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 6: Scatter CGPA vs Mock Interview ─────────────────────────────────
    st.subheader("CGPA vs Mock Interview Score (coloured by Placement)")
    sample = dff.sample(min(3000, len(dff)), random_state=42)
    fig = px.scatter(
        sample, x="cgpa", y="mock_interview_score",
        color="placement_status", opacity=0.6,
        color_discrete_map={"Placed": "#22c55e", "Not Placed": "#ef4444"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw Data Preview ──────────────────────────────────────────────────────
    with st.expander("🔎 View Raw Data Sample (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)
