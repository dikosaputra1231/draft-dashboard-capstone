"""
app.py - QLOP Market Insight Dashboard
======================================
Dashboard insight pasar kerja IT Indonesia sebagai fitur pendukung
aplikasi QLOP (NLP-driven Skill Gap Analysis Tool).

Jalankan: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QLOP – Market Insight Dashboard",
    page_icon="assets/logo.png" if False else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# DESIGN TOKENS  (light-mode, professional palette)
# ─────────────────────────────────────────────────────────────────
C_BG        = "#F8FAFC"   # page background
C_SURFACE   = "#FFFFFF"   # card / sidebar
C_BORDER    = "#E2E8F0"   # divider / border
C_TEXT      = "#0F172A"   # primary text
C_MUTED     = "#64748B"   # secondary text
C_BLUE      = "#2563EB"   # primary accent
C_VIOLET    = "#7C3AED"   # secondary accent
C_GREEN     = "#16A34A"
C_AMBER     = "#D97706"
C_RED       = "#DC2626"
C_BLUE_SOFT = "#DBEAFE"   # chip background
C_GREEN_SOFT= "#DCFCE7"
C_RED_SOFT  = "#FEE2E2"

PLOTLY_BASE = dict(
    paper_bgcolor=C_SURFACE,
    plot_bgcolor="#F1F5F9",
    font=dict(color=C_TEXT, family="Inter, system-ui, sans-serif", size=12),
    xaxis=dict(gridcolor=C_BORDER, color=C_MUTED, showline=False),
    yaxis=dict(gridcolor=C_BORDER, color=C_MUTED, showline=False),
    margin=dict(t=40, b=20, l=10, r=10),
)


def apply_theme(fig, height=400):
    fig.update_layout(height=height, **PLOTLY_BASE)
    return fig


# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Reset ────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: Inter, system-ui, sans-serif;
}}

/* ── Sidebar ───────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {C_SURFACE};
    border-right: 1px solid {C_BORDER};
}}
[data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}

/* ── Metric cards ──────────────────────────────────────────── */
[data-testid="metric-container"] {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}}
[data-testid="metric-container"] label {{
    color: {C_MUTED} !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: .03em;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {C_TEXT} !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricDelta"] {{ font-size: 12px !important; }}

/* ── Headings ──────────────────────────────────────────────── */
h1 {{ font-size: 1.75rem !important; font-weight: 800 !important;
       color: {C_TEXT} !important; margin-bottom: 0 !important; }}
h2 {{ font-size: 1.25rem !important; font-weight: 700 !important;
       color: {C_TEXT} !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 600 !important;
       color: {C_TEXT} !important; }}

/* ── Tabs ──────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {{
    background: {C_BORDER};
    border-radius: 8px;
    gap: 3px;
    padding: 3px;
}}
[data-baseweb="tab"] {{
    border-radius: 6px !important;
    color: {C_MUTED} !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    background: transparent !important;
}}
[aria-selected="true"] {{
    background: {C_SURFACE} !important;
    color: {C_BLUE} !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.10) !important;
}}

/* ── Divider ────────────────────────────────────────────────── */
hr {{ border: none; border-top: 1px solid {C_BORDER}; margin: 20px 0; }}

/* ── Pill chip ──────────────────────────────────────────────── */
.chip {{
    display: inline-block;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px 2px;
    line-height: 1.6;
}}
.chip-blue   {{ background:{C_BLUE_SOFT};  color:{C_BLUE};  }}
.chip-green  {{ background:{C_GREEN_SOFT}; color:{C_GREEN}; }}
.chip-red    {{ background:{C_RED_SOFT};   color:{C_RED};   }}
.chip-muted  {{ background:{C_BORDER};     color:{C_MUTED}; }}

/* ── Card ───────────────────────────────────────────────────── */
.card {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}}

/* ── Hero banner ─────────────────────────────────────────────── */
.hero {{
    background: linear-gradient(135deg, {C_BLUE} 0%, {C_VIOLET} 100%);
    border-radius: 14px;
    padding: 32px 36px;
    margin-bottom: 28px;
    color: white;
}}
.hero h1 {{ color: white !important; font-size: 2rem !important; margin: 0; }}
.hero p  {{ color: rgba(255,255,255,.85); margin: 8px 0 0; font-size: 15px; }}

/* ── Select / multiselect ───────────────────────────────────── */
[data-baseweb="select"] > div,
[data-baseweb="multi-select"] > div {{
    background: {C_SURFACE} !important;
    border-color: {C_BORDER} !important;
    color: {C_TEXT} !important;
}}

/* ── Button ─────────────────────────────────────────────────── */
.stButton > button {{
    background: {C_BLUE};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 9px 22px;
    font-weight: 600;
    font-size: 13px;
    transition: opacity .15s;
}}
.stButton > button:hover {{ opacity: .88; }}

/* ── Alerts ─────────────────────────────────────────────────── */
.stAlert {{ border-radius: 8px !important; }}

/* ── Caption / muted text ───────────────────────────────────── */
.stCaption, small, .muted {{ color: {C_MUTED} !important; font-size: 12px; }}

/* ── Progress bar ───────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {{
    background: {C_BLUE} !important;
}}

/* ── Sidebar nav label hidden ───────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] label {{
    font-weight: 600 !important; font-size: 13px !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# DATA LOADERS  (cached)
# ─────────────────────────────────────────────────────────────────
COURSERA_PATH = "data/raw/Coursera (1).csv"
JOBS_PATH     = "data/processed/MASTERED_DATA_FINAL_MODELING.csv"


@st.cache_data(ttl=3600)
def load_jobs():
    df = pd.read_csv(JOBS_PATH)
    df["skills_list"] = df["hard_skills"].apply(
        lambda x: [s.strip() for s in str(x).split(",") if s.strip()] if pd.notna(x) else []
    )
    df["postedAt"] = pd.to_datetime(df["postedAt"], dayfirst=True, errors="coerce")
    df["year"]  = df["postedAt"].dt.year
    df["month"] = df["postedAt"].dt.month
    return df


@st.cache_data(ttl=3600)
def load_coursera():
    df = pd.read_csv(COURSERA_PATH, sep=";", encoding="latin-1", header=1)
    df = df.dropna(subset=["Skills"]).copy()
    df["skills_list"] = df["Skills"].apply(
        lambda x: [s.strip() for s in str(x).split(";") if s.strip()]
    )
    df["skill_count"] = df["skills_list"].apply(len)
    df.rename(columns={
        "Name": "name", "Patners 1": "partner", "Url": "url",
        "Job category": "role_category", "Difficulty": "difficulty",
        "Duration": "duration"
    }, inplace=True)
    return df


@st.cache_data(ttl=3600)
def skill_counter(df):
    all_s: list = []
    df["skills_list"].apply(lambda l: all_s.extend(l))
    return Counter(all_s)


@st.cache_data(ttl=3600)
def build_role_profiles(df):
    all_s: list = []
    df["skills_list"].apply(lambda l: all_s.extend(l))
    freq = Counter(all_s)
    sel  = sorted([s for s, c in freq.items() if c >= 15])
    mlb  = MultiLabelBinarizer(classes=sel)
    X    = mlb.fit_transform(df["skills_list"])
    Xdf  = pd.DataFrame(X, columns=mlb.classes_)
    Xdf["role_label"] = df["role_label"].values
    return Xdf.groupby("role_label").mean(), mlb, sel


# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
df    = load_jobs()
df_c  = load_coursera()
j_ctr = skill_counter(df)
c_ctr = skill_counter(df_c)
role_profiles, mlb, sel_skills = build_role_profiles(df)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 20px;">
        <div style="font-size:36px;margin-bottom:6px;">📊</div>
        <div style="font-size:17px;font-weight:800;color:{C_TEXT};">QLOP</div>
        <div style="font-size:12px;color:{C_MUTED};margin-top:2px;">
            Market Insight Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Skill Demand", "Hiring Trends", "Course Supply", "A/B Testing"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Loker: {len(df):,} lowongan · {df['role_label'].nunique()} roles")
    st.caption(f"Course: {len(df_c):,} · Skill unik: {len(c_ctr):,}")
    st.caption("LinkedIn Indonesia 2022–2026 · Coursera")


# ─────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown(f"""
    <div class="hero">
        <h1>QLOP – Market Insight Dashboard</h1>
        <p>Wawasan makro pasar kerja IT Indonesia untuk mendukung sistem analisis skills gap QLOP</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ─────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Loker",   f"{len(df):,}")
    k2.metric("IT Roles",      df["role_label"].nunique())
    k3.metric("Skill Unik",    f"{len(j_ctr):,}")
    k4.metric("Total Course",  f"{len(df_c):,}")
    k5.metric("Remote Work",   f"{df['workRemoteAllowed'].fillna(0).astype(float).mean()*100:.1f}%")
    k6.metric("Rata Skill/Loker", f"{df['skill_count'].mean():.1f}")

    st.divider()

    # ── Row 1: Role distribution + Yearly trend ─────────────────
    col_a, col_b = st.columns([3, 2], gap="medium")

    with col_a:
        st.subheader("Distribusi Loker per Role IT")
        rc = df["role_label"].value_counts().reset_index()
        rc.columns = ["role", "count"]
        fig = px.bar(
            rc, x="count", y="role", orientation="h", text="count",
            color="count", color_continuous_scale="Blues",
        )
        fig.update_traces(textposition="outside", textfont_size=10, marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Jumlah Loker", yaxis_title="",
        )
        apply_theme(fig, height=620)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Tren Posting Loker (per Tahun)")
        yearly = (
            df[df["year"].between(2022, 2026)]
            .groupby("year").size().reset_index(name="count")
        )
        fig2 = px.area(
            yearly, x="year", y="count", markers=True,
            color_discrete_sequence=[C_BLUE],
        )
        fig2.update_traces(
            fill="tozeroy", fillcolor="rgba(37,99,235,.12)",
            line_width=2.5, marker_size=7,
        )
        fig2.update_layout(xaxis_title="Tahun", yaxis_title="Loker")
        apply_theme(fig2, height=240)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Employment Type")
        emp = df["employmentType"].value_counts().reset_index()
        emp.columns = ["type", "count"]
        fig3 = px.pie(
            emp, values="count", names="type", hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig3.update_traces(textposition="inside", textinfo="percent+label",
                           textfont_size=11)
        fig3.update_layout(showlegend=False)
        apply_theme(fig3, height=280)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── Row 2: Top skills demand vs supply ──────────────────────
    st.subheader("Top 10 Skill: Demand (Loker) vs Supply (Coursera)")
    ca, cb = st.columns(2, gap="medium")

    with ca:
        top10j = pd.DataFrame(j_ctr.most_common(10), columns=["skill", "count"])
        fig = px.bar(
            top10j, x="skill", y="count", text="count",
            color_discrete_sequence=[C_BLUE],
            title="Skill Paling Dicari di Loker",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(xaxis_tickangle=-35, xaxis_title="", yaxis_title="Frekuensi",
                          title_font_size=14)
        apply_theme(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        top10c = pd.DataFrame(c_ctr.most_common(10), columns=["skill", "count"])
        fig = px.bar(
            top10c, x="skill", y="count", text="count",
            color_discrete_sequence=[C_VIOLET],
            title="Skill Paling Banyak Diajarkan di Course",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(xaxis_tickangle=-35, xaxis_title="", yaxis_title="Frekuensi",
                          title_font_size=14)
        apply_theme(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 3: Seniority + Remote snapshot ──────────────────────
    cs, cr = st.columns(2, gap="medium")

    SEN_ORDER = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
    with cs:
        st.subheader("Distribusi Seniority Level")
        df_sen = df[df["seniorityLevel"].isin(SEN_ORDER)]
        sc = df_sen["seniorityLevel"].value_counts().reindex(SEN_ORDER).dropna().reset_index()
        sc.columns = ["level", "count"]
        fig = px.bar(
            sc, x="level", y="count", text="count",
            color="level",
            color_discrete_sequence=px.colors.sequential.Blues_r[:len(sc)],
        )
        fig.update_traces(textposition="outside", marker_line_width=0, showlegend=False)
        fig.update_layout(xaxis_title="", yaxis_title="Loker")
        apply_theme(fig, height=270)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.subheader("Remote Work – Proporsi per Role (Top 10)")
        df_rm = df.dropna(subset=["workRemoteAllowed"]).copy()
        df_rm["is_remote"] = df_rm["workRemoteAllowed"].astype(float).astype(bool)
        rr = (
            df_rm.groupby("role_label")
            .agg(total=("is_remote","count"), remote=("is_remote","sum"))
            .assign(pct=lambda x: (x.remote / x.total * 100).round(1))
            .nlargest(10, "pct").reset_index()
        )
        avg_remote = df_rm["is_remote"].mean() * 100
        fig = px.bar(
            rr, x="pct", y="role_label", orientation="h", text="pct",
            color="pct", color_continuous_scale="Greens",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        fig.add_vline(x=avg_remote, line_dash="dash", line_color=C_AMBER,
                      annotation_text=f"Avg {avg_remote:.1f}%", annotation_position="top right")
        fig.update_layout(yaxis={"categoryorder":"total ascending"},
                          xaxis_title="% Remote", yaxis_title="")
        apply_theme(fig, height=270)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: SKILL DEMAND
# ─────────────────────────────────────────────────────────────────
elif page == "Skill Demand":
    st.title("Skill Demand – Pasar Kerja IT")
    st.caption("Skill teknis yang paling banyak dicari berdasarkan lowongan kerja LinkedIn Indonesia")

    tab1, tab2, tab3 = st.tabs(["Top Skills Global", "Skills per Role", "Heatmap"])

    with tab1:
        r1, r2 = st.columns([1, 3], gap="medium")
        with r1:
            top_n     = st.slider("Top N Skills", 10, 50, 25)
            yr_filter = st.multiselect(
                "Tahun", [2022, 2023, 2024, 2025, 2026],
                default=[2022, 2023, 2024, 2025, 2026],
            )
        dff  = df[df["year"].isin(yr_filter)] if yr_filter else df
        s_all: list = []
        dff["skills_list"].apply(lambda l: s_all.extend(l))
        ctr_f = Counter(s_all)
        tsk   = pd.DataFrame(ctr_f.most_common(top_n), columns=["skill", "count"])

        with r2:
            fig = px.bar(
                tsk, x="count", y="skill", orientation="h", text="count",
                color="count", color_continuous_scale="Blues",
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            fig.update_layout(yaxis={"categoryorder":"total ascending"},
                              xaxis_title="Frekuensi", yaxis_title="")
            apply_theme(fig, max(420, top_n * 22))
            st.plotly_chart(fig, use_container_width=True)

        # Trend line
        st.subheader("Tren Top 8 Skill per Tahun")
        top8 = [s for s, _ in ctr_f.most_common(8)]
        rows = []
        for _, row in df.dropna(subset=["year"]).iterrows():
            yr = int(row["year"])
            if 2022 <= yr <= 2026:
                for sk in row["skills_list"]:
                    if sk in top8:
                        rows.append({"Year": yr, "Skill": sk})
        if rows:
            tdf  = pd.DataFrame(rows)
            tgrp = tdf.groupby(["Year","Skill"]).size().reset_index(name="Count")
            fig2 = px.line(
                tgrp, x="Year", y="Count", color="Skill", markers=True,
                line_shape="spline",
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            fig2.update_traces(line_width=2.5)
            apply_theme(fig2, 380)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        rs = st.selectbox("Pilih Role IT", sorted(df["role_label"].unique()))
        ns = st.slider("Jumlah Skill", 5, 20, 12)
        dfr  = df[df["role_label"] == rs]
        rs_all: list = []
        dfr["skills_list"].apply(lambda l: rs_all.extend(l))
        rctr  = Counter(rs_all)
        tprol = pd.DataFrame(rctr.most_common(ns), columns=["skill","count"])

        c1, c2 = st.columns([2, 1], gap="medium")
        with c1:
            if not tprol.empty:
                fig = px.bar(
                    tprol, x="count", y="skill", orientation="h", text="count",
                    color="count", color_continuous_scale="Blues",
                    title=f"Top {ns} Skill – {rs} ({len(dfr):,} loker)",
                )
                fig.update_traces(textposition="outside", marker_line_width=0)
                fig.update_coloraxes(showscale=False)
                fig.update_layout(yaxis={"categoryorder":"total ascending"},
                                  xaxis_title="Frekuensi", yaxis_title="")
                apply_theme(fig, max(320, ns * 36))
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown(f"**Profil: {rs}**")
            st.metric("Total Loker",     f"{len(dfr):,}")
            st.metric("Avg Skill/Loker", f"{dfr['skill_count'].mean():.1f}")
            st.metric("Remote Work",
                      f"{dfr['workRemoteAllowed'].fillna(0).astype(float).mean()*100:.1f}%")
            st.metric("Avg Pelamar",     f"{dfr['applicantsCount'].mean():.0f}")
            st.markdown("**Skill utama:**")
            for s, c in rctr.most_common(5):
                pct = c / len(dfr) * 100
                st.markdown(
                    f"<span class='chip chip-blue'>{s} &nbsp;{pct:.0f}%</span>",
                    unsafe_allow_html=True,
                )

    with tab3:
        st.subheader("Heatmap: % Kemunculan Skill per Role")
        nh = st.slider("Jumlah top skill", 10, 30, 20)
        th_skills = [s for s, _ in j_ctr.most_common(nh)]
        th_roles  = df["role_label"].value_counts().head(15).index.tolist()
        hm = {}
        for role in th_roles:
            dfr2 = df[df["role_label"] == role]
            tot  = len(dfr2)
            hm[role] = {
                sk: round(dfr2["skills_list"].apply(lambda l: sk in l).sum() / tot * 100, 1)
                for sk in th_skills
            }
        hm_df = pd.DataFrame(hm).T
        fig = px.imshow(
            hm_df, color_continuous_scale="Blues",
            aspect="auto", text_auto=".0f",
        )
        fig.update_layout(coloraxis_colorbar_title="%")
        fig.update_xaxes(tickangle=-40)
        apply_theme(fig, 520)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: HIRING TRENDS
# ─────────────────────────────────────────────────────────────────
elif page == "Hiring Trends":
    st.title("Hiring Trends")
    st.caption("Tren perekrutan, kompetisi, seniority, dan remote work di IT Indonesia")

    tab1, tab2, tab3 = st.tabs(["Kompetisi", "Remote Work", "Seniority & Employment"])

    with tab1:
        st.subheader("Kompetisi per Role – Rata-rata Jumlah Pelamar")
        comp = (
            df.groupby("role_label")
            .agg(avg_applicants=("applicantsCount","mean"),
                 total_jobs=("role_label","count"))
            .round(1).sort_values("avg_applicants", ascending=False).reset_index()
        )
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            fig = px.bar(
                comp, x="avg_applicants", y="role_label", orientation="h",
                text="avg_applicants",
                color="avg_applicants", color_continuous_scale="RdYlGn_r",
            )
            fig.update_traces(texttemplate="%{text:.0f}", textposition="outside",
                              marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            fig.update_layout(yaxis={"categoryorder":"total ascending"},
                              xaxis_title="Rata-rata Pelamar", yaxis_title="")
            apply_theme(fig, 620)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.scatter(
                comp, x="total_jobs", y="avg_applicants",
                size="total_jobs", color="avg_applicants",
                color_continuous_scale="RdYlGn_r", text="role_label",
                title="Loker vs Kompetisi",
            )
            fig2.update_traces(textposition="top center", textfont_size=9)
            fig2.update_coloraxes(showscale=False)
            fig2.update_layout(xaxis_title="Total Loker", yaxis_title="Avg Pelamar")
            apply_theme(fig2, 620)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Adopsi Remote Work per Role IT")
        df_rm = df.dropna(subset=["workRemoteAllowed"]).copy()
        df_rm["is_remote"] = df_rm["workRemoteAllowed"].astype(float).astype(bool)
        rr2 = (
            df_rm.groupby("role_label")
            .agg(total=("is_remote","count"), remote=("is_remote","sum"))
            .assign(pct=lambda x: (x.remote / x.total * 100).round(1))
            .sort_values("pct", ascending=False).reset_index()
        )
        avg_r = df_rm["is_remote"].mean() * 100
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            fig = px.bar(
                rr2, x="pct", y="role_label", orientation="h", text="pct",
                color="pct", color_continuous_scale="Greens",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            fig.add_vline(x=avg_r, line_dash="dash", line_color=C_AMBER,
                          annotation_text=f"Avg {avg_r:.1f}%")
            fig.update_layout(yaxis={"categoryorder":"total ascending"},
                              xaxis_title="% Remote", yaxis_title="")
            apply_theme(fig, 560)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            yes = df_rm["is_remote"].sum(); no = (~df_rm["is_remote"]).sum()
            fig2 = px.pie(
                values=[yes, no], names=["Remote","On-site"], hole=0.55,
                color_discrete_sequence=[C_GREEN, C_BORDER],
            )
            fig2.update_traces(textposition="inside", textinfo="percent+label",
                               textfont_size=12)
            apply_theme(fig2, 280)
            st.plotly_chart(fig2, use_container_width=True)
            st.metric("Rata-rata Remote",      f"{avg_r:.1f}%")
            st.metric("Paling Remote-friendly", rr2.iloc[0]["role_label"])

    with tab3:
        st.subheader("Distribusi Seniority Level")
        SEN_ORDER = ["Entry level","Associate","Mid-Senior level","Director","Executive"]
        df_sen = df[df["seniorityLevel"].isin(SEN_ORDER)].copy()
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            sc = df_sen["seniorityLevel"].value_counts().reset_index()
            sc.columns = ["level","count"]
            fig = px.pie(sc, values="count", names="level", hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              textfont_size=11)
            apply_theme(fig, 320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sr = df_sen.groupby(["role_label","seniorityLevel"]).size().unstack(fill_value=0)
            sp = sr.div(sr.sum(axis=1), axis=0) * 100
            top8 = df["role_label"].value_counts().head(8).index
            sp   = sp.reindex(top8)
            ex   = [c for c in SEN_ORDER if c in sp.columns]
            fig2 = px.bar(sp[ex].reset_index(), y="role_label",
                          x=ex, barmode="stack", orientation="h",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(xaxis_title="%", yaxis_title="", legend_title="Level")
            apply_theme(fig2, 320)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Employment Type")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            em = df["employmentType"].value_counts().reset_index()
            em.columns = ["type","count"]
            fig = px.pie(em, values="count", names="type", hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              textfont_size=11)
            apply_theme(fig, 280)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            top6 = df["role_label"].value_counts().head(6).index
            er   = df[df["role_label"].isin(top6)].groupby(
                    ["role_label","employmentType"]).size().unstack(fill_value=0)
            ep   = er.div(er.sum(axis=1), axis=0) * 100
            fig2 = px.bar(ep.reset_index(), y="role_label",
                          x=[c for c in ep.columns], barmode="stack", orientation="h",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(xaxis_title="%", yaxis_title="", legend_title="Tipe")
            apply_theme(fig2, 280)
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: COURSE SUPPLY
# ─────────────────────────────────────────────────────────────────
elif page == "Course Supply":
    st.title("Course Supply – Analisis Coursera")
    st.caption("Ketersediaan course untuk menutup kesenjangan skill di pasar kerja IT")

    tab1, tab2, tab3 = st.tabs(["Overview Course", "Supply vs Demand", "Course per Role"])

    with tab1:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Course",    f"{len(df_c):,}")
        k2.metric("Role Tercakup",   df_c["role_category"].nunique())
        k3.metric("Skill Unik",      f"{len(c_ctr):,}")
        k4.metric("Avg Skill/Course",f"{df_c['skill_count'].mean():.1f}")

        st.divider()
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.subheader("Distribusi Course per Role (Top 15)")
            cc = df_c["role_category"].value_counts().head(15).reset_index()
            cc.columns = ["role","count"]
            fig = px.bar(
                cc, x="count", y="role", orientation="h", text="count",
                color="count", color_continuous_scale="Purples",
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            fig.update_layout(yaxis={"categoryorder":"total ascending"},
                              xaxis_title="Jumlah Course", yaxis_title="")
            apply_theme(fig, 440)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Tingkat Kesulitan")
            di = df_c["difficulty"].value_counts().reset_index()
            di.columns = ["level","count"]
            fig2 = px.pie(
                di, values="count", names="level", hole=0.55,
                color_discrete_sequence=[C_GREEN, C_BLUE, C_AMBER, C_RED],
            )
            fig2.update_traces(textposition="inside", textinfo="percent+label",
                               textfont_size=12)
            apply_theme(fig2, 200)
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Durasi Course")
            du = df_c["duration"].value_counts().reset_index()
            du.columns = ["durasi","count"]
            du["durasi"] = du["durasi"].str.replace("_"," ").str.title()
            fig3 = px.bar(
                du, x="durasi", y="count", text="count",
                color="count", color_continuous_scale="Blues",
            )
            fig3.update_traces(textposition="outside", marker_line_width=0)
            fig3.update_coloraxes(showscale=False)
            fig3.update_layout(xaxis_tickangle=-30, xaxis_title="", yaxis_title="Jumlah")
            apply_theme(fig3, 220)
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.subheader("Skill Supply (Coursera) vs Demand (Loker)")
        st.info(
            "Skill dengan **demand tinggi** namun **supply rendah** adalah "
            "prioritas utama pembelajaran dan pengembangan course."
        )
        TOP_N = st.slider("Jumlah skill top demand", 15, 50, 30)
        top_d = dict(j_ctr.most_common(TOP_N))
        top_s = dict(c_ctr.most_common(300))

        gdf = pd.DataFrame([
            {"skill": sk, "demand": dm, "supply": top_s.get(sk, 0),
             "gap_ratio": dm / max(top_s.get(sk, 0), 1)}
            for sk, dm in top_d.items()
        ]).sort_values("gap_ratio", ascending=False)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            # Grouped bar
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Demand (Loker)",   y=gdf["skill"], x=gdf["demand"],
                orientation="h", marker_color=C_BLUE, marker_line_width=0,
            ))
            fig.add_trace(go.Bar(
                name="Supply (Course)",  y=gdf["skill"], x=gdf["supply"],
                orientation="h", marker_color=C_VIOLET, marker_line_width=0,
            ))
            fig.update_layout(
                barmode="group",
                yaxis={"categoryorder":"array",
                       "categoryarray": gdf.sort_values("demand")["skill"].tolist()},
                xaxis_title="Frekuensi", yaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            apply_theme(fig, max(500, TOP_N * 25))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Gap Ratio Terbesar")
            st.caption("Demand/Supply ratio – semakin tinggi semakin kurang terlayani")
            g15 = gdf.head(15)
            fig2 = px.bar(
                g15, x="gap_ratio", y="skill", orientation="h",
                text=g15["gap_ratio"].round(1),
                color="gap_ratio", color_continuous_scale="Reds",
            )
            fig2.update_traces(texttemplate="%{text:.1f}x", textposition="outside",
                               marker_line_width=0)
            fig2.update_coloraxes(showscale=False)
            fig2.update_layout(
                yaxis={"categoryorder":"total ascending"},
                xaxis_title="Demand / Supply Ratio", yaxis_title="",
            )
            apply_theme(fig2, max(380, 15 * 28))
            st.plotly_chart(fig2, use_container_width=True)

        # Categorisation chips
        st.divider()
        st.subheader("Kategorisasi Skill")
        hi_d = set(s for s, _ in j_ctr.most_common(30))
        hi_s = set(s for s, _ in c_ctr.most_common(30))
        cm, cg, cs_col = st.columns(3, gap="medium")
        with cm:
            st.markdown("**Terlayani Baik** — demand & supply sama-sama tinggi")
            for s in sorted(hi_d & hi_s):
                st.markdown(f"<span class='chip chip-green'>{s}</span>",
                            unsafe_allow_html=True)
        with cg:
            st.markdown("**Kurang Terlayani** — demand tinggi, supply rendah")
            for s in sorted(hi_d - hi_s):
                st.markdown(f"<span class='chip chip-red'>{s}</span>",
                            unsafe_allow_html=True)
        with cs_col:
            st.markdown("**Oversupplied** — banyak course, demand lebih rendah")
            for s in sorted(hi_s - hi_d):
                st.markdown(f"<span class='chip chip-muted'>{s}</span>",
                            unsafe_allow_html=True)

    with tab3:
        st.subheader("Course yang Tersedia per Role IT")
        rs3 = st.selectbox("Pilih Role", sorted(df["role_label"].unique()), key="cs_role")
        df_cr = df_c[df_c["role_category"] == rs3]
        if df_cr.empty:
            df_cr = df_c[df_c["role_category"].str.contains(
                rs3.split()[0], case=False, na=False)]

        c1, c2 = st.columns([2, 1], gap="medium")
        with c1:
            if not df_cr.empty:
                st.success(f"Ditemukan **{len(df_cr)} course** untuk role **{rs3}**")
                for _, row in df_cr.head(15).iterrows():
                    diff_col = {
                        "BEGINNER": C_GREEN, "INTERMEDIATE": C_AMBER,
                        "ADVANCED": C_RED,   "MIXED": C_BLUE,
                    }.get(str(row.get("difficulty","")), C_MUTED)
                    dur_str  = str(row.get("duration","")).replace("_"," ").title()
                    skp      = row["skills_list"][:6]
                    more     = f" +{len(row['skills_list'])-6} lainnya" if len(row["skills_list"]) > 6 else ""
                    st.markdown(f"""
                    <div class="card">
                        <div style="font-size:14px;font-weight:700;">
                            <a href="{row['url']}" target="_blank"
                               style="color:{C_BLUE};text-decoration:none;">
                                {row['name']}
                            </a>
                        </div>
                        <div style="font-size:12px;color:{C_MUTED};margin:4px 0;">
                            {row['partner']}
                            &nbsp;·&nbsp;
                            <span style="color:{diff_col};font-weight:600;">
                                {row.get('difficulty','')}
                            </span>
                            &nbsp;·&nbsp; {dur_str}
                        </div>
                        <div style="font-size:12px;color:{C_TEXT};">
                            {", ".join(skp)}{more}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"Belum ada course Coursera yang ter-mapping ke role '{rs3}'.")

        with c2:
            j_rs = []
            df[df["role_label"] == rs3]["skills_list"].apply(lambda l: j_rs.extend(l))
            jrc  = Counter(j_rs)
            top_j = set(s for s, _ in jrc.most_common(10))

            c_rs: list = []
            df_cr["skills_list"].apply(lambda l: c_rs.extend(l))
            crc  = Counter(c_rs)
            top_c = set(s for s, _ in crc.most_common(10))

            cov = top_j & top_c; unc = top_j - top_c
            cov_pct = len(cov) / max(len(top_j), 1) * 100
            st.markdown(f"**Skill Coverage – {rs3}**")
            st.metric("Skill terlayani", f"{len(cov)} / {len(top_j)}")
            st.progress(cov_pct / 100,
                        text=f"Coverage {cov_pct:.0f}%")
            if cov:
                st.markdown("**Tercakup:**")
                for s in sorted(cov):
                    st.markdown(f"<span class='chip chip-green'>{s}</span>",
                                unsafe_allow_html=True)
            if unc:
                st.markdown("**Belum tercakup:**")
                for s in sorted(unc):
                    st.markdown(f"<span class='chip chip-red'>{s}</span>",
                                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: A/B TESTING
# ─────────────────────────────────────────────────────────────────
elif page == "A/B Testing":
    st.title("A/B Testing – Strategi Rekomendasi Role")
    st.caption("Eksperimen validasi: Top-3 Heuristik vs Cosine Similarity untuk mapping skill CV ke role")

    st.info(
        "**Konteks QLOP:** AI Engine perlu memilih strategi terbaik untuk memetakan "
        "skill yang diekstrak dari CV ke role IT. Eksperimen ini membandingkan dua pendekatan.\n\n"
        "**H₀:** Tidak ada perbedaan signifikan · "
        "**H₁:** Strategi B (Cosine Similarity) lebih akurat · "
        "**α = 0.05** · Metrik: Precision@1"
    )

    @st.cache_data(ttl=7200)
    def run_ab():
        from sklearn.model_selection import train_test_split
        from scipy import stats as sp

        dfab = pd.read_csv(JOBS_PATH)
        dfab["skills_list"] = dfab["hard_skills"].apply(
            lambda x: [s.strip() for s in str(x).split(",") if s.strip()] if pd.notna(x) else []
        )
        s_all: list = []
        dfab["skills_list"].apply(lambda l: s_all.extend(l))
        freq = Counter(s_all)
        sel2 = sorted([s for s, c in freq.items() if c >= 20])

        mlb2  = MultiLabelBinarizer(classes=sel2)
        X_all = mlb2.fit_transform(dfab["skills_list"])

        tr, te = train_test_split(dfab, test_size=0.2, random_state=42,
                                  stratify=dfab["role_label"])
        Xtr = mlb2.transform(tr["skills_list"])
        Xte = mlb2.transform(te["skills_list"])
        yte = te["role_label"].values

        trm = pd.DataFrame(Xtr, columns=mlb2.classes_)
        trm["role_label"] = tr["role_label"].values
        rp  = trm.groupby("role_label").mean()
        ro  = rp.index.tolist()

        # Strategy A
        rt3 = {r: set(rp.loc[r].nlargest(3).index) for r in ro}
        def pred_a(skills):
            best, bs = None, -1
            for r, t3 in rt3.items():
                sc = len(set(skills) & t3)
                if sc > bs: bs, best = sc, r
            return best or "Software Engineer"
        pA = [pred_a(te["skills_list"].iloc[i]) for i in range(len(te))]

        # Strategy B
        pB = [ro[np.argmax(cosine_similarity(Xte[i:i+1], rp.values)[0])]
              for i in range(len(Xte))]

        aA = np.mean(np.array(pA) == yte)
        aB = np.mean(np.array(pB) == yte)
        cA = (np.array(pA) == yte).astype(int)
        cB = (np.array(pB) == yte).astype(int)

        b  = ((cA==1)&(cB==0)).sum(); c = ((cA==0)&(cB==1)).sum()
        st2 = (abs(b-c)-1)**2/(b+c) if (b+c) > 0 else 0
        pv  = sp.chi2.sf(st2, df=1)

        np.random.seed(42)
        diffs = np.array([
            cB[np.random.choice(len(cA), len(cA), replace=True)].mean() -
            cA[np.random.choice(len(cA), len(cA), replace=True)].mean()
            for _ in range(3000)
        ])
        return aA, aB, pv, diffs, len(yte)

    with st.spinner("Menjalankan eksperimen A/B (15–30 detik)..."):
        aA, aB, pv, boot, n_te = run_ab()

    ci_lo = np.percentile(boot, 2.5) * 100
    ci_hi = np.percentile(boot, 97.5) * 100
    mean_d = boot.mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Strategi A – Precision@1", f"{aA*100:.2f}%", help="Top-3 Heuristik")
    k2.metric("Strategi B – Precision@1", f"{aB*100:.2f}%",
              delta=f"{(aB-aA)*100:+.2f}%", help="Cosine Similarity")
    k3.metric("McNemar p-value", f"{pv:.4f}")
    k4.metric("Keputusan",
              "Tolak H₀" if pv < 0.05 else "Gagal Tolak H₀",
              delta="Signifikan" if pv < 0.05 else "Tidak Signifikan",
              delta_color="normal" if pv < 0.05 else "off")

    st.divider()
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.subheader("Perbandingan Akurasi")
        fig = go.Figure(go.Bar(
            x=["Strategi A\n(Top-3 Heuristik)", "Strategi B\n(Cosine Similarity)"],
            y=[aA*100, aB*100],
            text=[f"{aA*100:.2f}%", f"{aB*100:.2f}%"],
            textposition="outside",
            marker_color=[C_MUTED, C_BLUE],
            marker_line_width=0,
        ))
        fig.update_layout(yaxis_range=[0, max(aA, aB)*128], yaxis_title="Precision@1 (%)")
        apply_theme(fig, 340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Bootstrap Distribution (Selisih B − A)")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=boot*100, nbinsx=50,
            marker_color=C_BLUE, opacity=0.75, name="Bootstrap",
        ))
        fig2.add_vline(x=mean_d, line_dash="dash", line_color=C_RED,
                       annotation_text=f"Mean: {mean_d:+.2f}%")
        fig2.add_vline(x=ci_lo,  line_dash="dot",  line_color=C_AMBER,
                       annotation_text=f"{ci_lo:+.1f}%")
        fig2.add_vline(x=ci_hi,  line_dash="dot",  line_color=C_AMBER,
                       annotation_text=f"{ci_hi:+.1f}%")
        fig2.add_vline(x=0, line_color="#94A3B8", opacity=0.5)
        fig2.update_layout(xaxis_title="Selisih Akurasi B−A (%)",
                           yaxis_title="Frekuensi")
        apply_theme(fig2, 340)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"95% CI: [{ci_lo:+.2f}%, {ci_hi:+.2f}%] — {n_te:,} sampel test")

    st.divider()
    st.subheader("Kesimpulan & Rekomendasi untuk QLOP")
    if pv < 0.05:
        winner = "Strategi B (Cosine Similarity)" if aB > aA else "Strategi A (Top-3)"
        st.success(
            f"**H₀ ditolak** (p = {pv:.4f} < 0.05)  \n"
            f"Terdapat perbedaan **signifikan secara statistik**.  \n"
            f"**{winner}** terbukti lebih akurat dengan keyakinan 95%.  \n\n"
            f"Selisih: {(aB-aA)*100:+.2f}% &nbsp;(95% CI: [{ci_lo:+.2f}%, {ci_hi:+.2f}%])"
        )
    else:
        st.warning(
            f"**Gagal tolak H₀** (p = {pv:.4f} ≥ 0.05)  \n"
            f"Tidak ada perbedaan signifikan secara statistik.  \n\n"
            f"**Rekomendasi untuk QLOP:** Gunakan **Strategi B (Cosine Similarity)** "
            f"karena lebih scalable — memanfaatkan seluruh dimensi skill vektor, "
            f"bukan hanya top-3."
        )
