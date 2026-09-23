"""
Streamlit main entry: placement_predictor/frontend/app.py
Run: streamlit run placement_predictor/frontend/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/320px-IBM_logo.svg.png",
    width=100,
)
st.sidebar.title("🎓 Placement Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 EDA Dashboard", "🔮 Single Prediction",
     "📁 Batch Prediction", "📈 Model Performance"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Backend:** Flask REST API (port 5000)  \n"
    "**Model:** Ensemble ML (RF / GB / LR)  \n"
    "**Dataset:** 100,000 students"
)

# ── Route to pages ────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from pages.home import show
    show()
elif page == "📊 EDA Dashboard":
    from pages.eda import show
    show()
elif page == "🔮 Single Prediction":
    from pages.predict import show
    show()
elif page == "📁 Batch Prediction":
    from pages.batch import show
    show()
elif page == "📈 Model Performance":
    from pages.performance import show
    show()
