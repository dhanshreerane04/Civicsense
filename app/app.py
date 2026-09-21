import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.predictor import predict_category, predict_batch, MODEL_LOAD_ERROR, device, keyword_rules  # noqa: E402


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="CivicSense",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Session State
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: complaint, category, confidence, method, timestamp

if "complaint_text" not in st.session_state:
    st.session_state.complaint_text = ""


CATEGORIES = sorted(keyword_rules.keys())

SAMPLE_COMPLAINTS = [
    "Bijli meter kharab hai, poora mohalla andhere mein hai",
    "Water supply has been cut off in our area for 3 days",
    "Local school has no teachers for the last two months",
    "Road pe bada pothole hai, accident ho sakta hai"
]


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.10), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(20, 184, 166, 0.10), transparent 30%),
            #f8fafc;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    .hero {
        padding: 2.5rem 2.5rem 2.2rem 2.5rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #172554 0%, #1e3a8a 45%, #0f766e 100%);
        color: white;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.5rem;
    }

    .hero-icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .hero-title { font-size: 3rem; font-weight: 800; letter-spacing: -1px; margin-bottom: 0.3rem; }
    .hero-subtitle { font-size: 1.1rem; opacity: 0.88; max-width: 700px; line-height: 1.6; }

    .hero-badge {
        display: inline-block;
        margin-top: 1.1rem;
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.22);
        font-size: 0.82rem;
        margin-right: 0.5rem;
    }

    .section-title { font-size: 1.4rem; font-weight: 750; color: #172554; margin-top: 0.5rem; margin-bottom: 0.6rem; }
    .section-description { color: #64748b; margin-bottom: 1rem; }

    .input-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow: 0 10px 30px rgba(15,23,42,0.06);
    }

    .result-card {
        margin-top: 1.3rem;
        padding: 1.7rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
        border: 1px solid #bbf7d0;
        box-shadow: 0 12px 30px rgba(22, 101, 52, 0.08);
    }

    .result-label { color: #166534; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; }
    .result-category { color: #14532d; font-size: 1.7rem; font-weight: 800; margin-top: 0.3rem; }

    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.3rem;
        height: 100%;
        box-shadow: 0 8px 24px rgba(15,23,42,0.05);
    }

    .info-icon { font-size: 1.7rem; }
    .info-title { font-weight: 700; color: #172554; margin-top: 0.4rem; }
    .info-text { color: #64748b; font-size: 0.88rem; line-height: 1.5; }

    .category-pill {
        display: inline-block;
        padding: 0.4rem 0.7rem;
        margin: 0.2rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        border: 1px solid #e0e7ff;
        font-size: 0.8rem;
    }

    .keyword-pill {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        margin: 0.15rem;
        border-radius: 999px;
        background: #fff7ed;
        color: #9a3412;
        border: 1px solid #fed7aa;
        font-size: 0.78rem;
    }

    .confidence-container { margin-top: 1rem; }
    .confidence-label { display: flex; justify-content: space-between; color: #475569; font-size: 0.88rem; margin-bottom: 0.3rem; }
    .confidence-bar { width: 100%; height: 10px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
    .confidence-fill { height: 100%; background: linear-gradient(90deg, #2563eb, #14b8a6); border-radius: 999px; }

    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 6px 18px rgba(15,23,42,0.05);
    }
    .stat-number { font-size: 1.6rem; font-weight: 800; color: #172554; }
    .stat-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }

    .custom-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Hero
# ============================================================

status_badge = "🟢 Model Ready" if MODEL_LOAD_ERROR is None else "🔴 Model Not Loaded"
device_label = f"⚙️ Running on {str(device).upper()}" if device is not None else ""

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-icon">🏛️</div>
        <div class="hero-title">CivicSense</div>
        <div class="hero-subtitle">
            AI-powered citizen grievance classification for
            Hindi, Hinglish and English complaints.
        </div>
        <div>
            <span class="hero-badge">🤖 IndicBERT + Hybrid NLP</span>
            <span class="hero-badge">{status_badge}</span>
            {f'<span class="hero-badge">{device_label}</span>' if device_label else ''}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if MODEL_LOAD_ERROR is not None:
    st.error(
        f"⚠️ The model could not be loaded, so classification is disabled.\n\n"
        f"**Details:** {MODEL_LOAD_ERROR}\n\n"
        f"Check that your model folder exists at the expected path "
        f"(`models/civicsense_indicbert`) relative to your project root."
    )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### 📊 Session Stats")

    total = len(st.session_state.history)
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-number">{total}</div>
            <div class="stat-label">Complaints Classified</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    if total > 0:
        hist_df = pd.DataFrame(st.session_state.history)
        top_cat = hist_df["category"].value_counts().idxmax()
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number" style="font-size:1.1rem;">{top_cat}</div>
                <div class="stat-label">Most Common Category</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    if st.button("🗑️ Clear Session History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.caption("CivicSense · IndicBERT + Hybrid Classification")


# ============================================================
# Tabs
# ============================================================

tab_single, tab_bulk, tab_analytics, tab_about = st.tabs(
    ["🔍 Single Complaint", "📁 Bulk Upload (CSV)", "📈 Analytics", "ℹ️ About"]
)


# ------------------------------------------------------------
# TAB 1 — Single Complaint
# ------------------------------------------------------------

with tab_single:

    st.markdown('<div class="section-title">📝 Submit a Grievance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">Describe the civic issue and CivicSense will identify '
        'the most relevant public-service category.</div>',
        unsafe_allow_html=True
    )

    st.markdown("**Try a sample complaint:**")
    sample_cols = st.columns(len(SAMPLE_COMPLAINTS))
    for i, sample in enumerate(SAMPLE_COMPLAINTS):
        with sample_cols[i]:
            if st.button(sample[:28] + ("…" if len(sample) > 28 else ""), key=f"sample_{i}", use_container_width=True):
                st.session_state.complaint_text = sample
                st.rerun()

    complaint = st.text_area(
        "Complaint",
        value=st.session_state.complaint_text,
        label_visibility="collapsed",
        height=150,
        key="complaint_input"
    )

    predict_clicked = st.button(
        "🔍  Classify Grievance",
        use_container_width=True,
        type="primary",
        disabled=(MODEL_LOAD_ERROR is not None)
    )

    if predict_clicked:
        if not complaint.strip():
            st.warning("Please enter a complaint before classifying.")
        else:
            with st.spinner("Analyzing grievance..."):
                result = predict_category(complaint)

            category = result["category"]
            confidence = result["confidence"]
            method = result["method"]
            matched_keywords = result["matched_keywords"]
            confidence_percent = confidence * 100

            st.session_state.history.append({
                "complaint": complaint,
                "category": category,
                "confidence": confidence,
                "method": method,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Category</div>
                    <div class="result-category">{category}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🤖 IndicBERT Confidence", f"{confidence_percent:.2f}%")
            with col2:
                st.metric("⚙️ Prediction Method", method)

            safe_width = min(max(confidence_percent, 0), 100)
            st.markdown(
                f"""
                <div class="confidence-container">
                    <div class="confidence-label">
                        <span>Model confidence</span>
                        <span>{confidence_percent:.2f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {safe_width}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if matched_keywords:
                st.markdown("**Matched keywords:**")
                pills = "".join(f'<span class="keyword-pill">{kw}</span>' for kw in matched_keywords)
                st.markdown(pills, unsafe_allow_html=True)


# ------------------------------------------------------------
# TAB 2 — Bulk Upload (CSV)
# ------------------------------------------------------------

with tab_bulk:

    st.markdown('<div class="section-title">📁 Bulk Classify from CSV</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">Upload a CSV file containing complaints, pick the column '
        'with the complaint text, and classify them all at once.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            df = None

        if df is not None and len(df) > 0:
            st.markdown(f"**Preview** ({len(df)} rows found)")
            st.dataframe(df.head(5), use_container_width=True)

            text_column = st.selectbox("Which column contains the complaint text?", options=df.columns)

            run_bulk = st.button(
                "🚀 Classify All Rows",
                use_container_width=True,
                type="primary",
                disabled=(MODEL_LOAD_ERROR is not None)
            )

            if run_bulk:
                complaints = df[text_column].astype(str).tolist()
                progress_bar = st.progress(0, text="Starting classification...")

                def update_progress(done, total):
                    progress_bar.progress(done / total, text=f"Classified {done}/{total} complaints")

                results = predict_batch(complaints, progress_callback=update_progress)
                progress_bar.empty()

                results_df = pd.DataFrame(results)
                results_df["confidence"] = (results_df["confidence"] * 100).round(2)
                results_df["matched_keywords"] = results_df["matched_keywords"].apply(lambda kws: ", ".join(kws))

                for r in results:
                    st.session_state.history.append({
                        "complaint": r["complaint"],
                        "category": r["category"],
                        "confidence": r["confidence"],
                        "method": r["method"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                st.success(f"Classified {len(results_df)} complaints.")
                st.dataframe(results_df, use_container_width=True)

                st.markdown("**Category distribution:**")
                st.bar_chart(results_df["category"].value_counts())

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv_bytes,
                    file_name="civicsense_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# ------------------------------------------------------------
# TAB 3 — Analytics
# ------------------------------------------------------------

with tab_analytics:

    st.markdown('<div class="section-title">📈 Session Analytics</div>', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("No complaints classified yet in this session. Try the Single Complaint or Bulk Upload tabs.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df["confidence_pct"] = (hist_df["confidence"] * 100).round(2)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""<div class="stat-card"><div class="stat-number">{len(hist_df)}</div>
                <div class="stat-label">Total Classified</div></div>""",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""<div class="stat-card"><div class="stat-number">{hist_df['confidence_pct'].mean():.1f}%</div>
                <div class="stat-label">Avg Confidence</div></div>""",
                unsafe_allow_html=True
            )
        with col3:
            hybrid_pct = (hist_df["method"].str.contains("Hybrid").mean() * 100)
            st.markdown(
                f"""<div class="stat-card"><div class="stat-number">{hybrid_pct:.0f}%</div>
                <div class="stat-label">Used Keyword Hybrid</div></div>""",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Category distribution:**")
        st.bar_chart(hist_df["category"].value_counts())

        st.markdown("**Recent classifications:**")
        st.dataframe(
            hist_df[["timestamp", "complaint", "category", "confidence_pct", "method"]]
            .rename(columns={"confidence_pct": "confidence (%)"})
            .sort_values("timestamp", ascending=False),
            use_container_width=True
        )


# ------------------------------------------------------------
# TAB 4 — About
# ------------------------------------------------------------

with tab_about:

    st.markdown('<div class="section-title">⚡ How CivicSense Works</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """<div class="info-card"><div class="info-icon">📝</div>
            <div class="info-title">1. Enter Complaint</div>
            <div class="info-text">Submit a grievance in Hindi, Hinglish or English — one at a time
            or in bulk via CSV.</div></div>""",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """<div class="info-card"><div class="info-icon">🧠</div>
            <div class="info-title">2. NLP Analysis</div>
            <div class="info-text">IndicBERT analyzes the complaint and identifies the relevant
            intent, backed by domain keyword rules.</div></div>""",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """<div class="info-card"><div class="info-icon">🎯</div>
            <div class="info-title">3. Classification</div>
            <div class="info-text">The grievance is assigned to one of 14 civic-service
            categories with a confidence score.</div></div>""",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div class="section-title">📂 Supported Categories</div>', unsafe_allow_html=True)

    category_html = "".join(f'<span class="category-pill">{c}</span>' for c in CATEGORIES)
    st.markdown(category_html, unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="custom-footer">
        <b>CivicSense</b> · Multilingual Citizen Grievance Classification
        <br>
        NLP · IndicBERT · Hybrid Classification · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)