"""
app.py — QLOP Market Insight Dashboard
=======================================
Supports light mode, dark mode, and system preference automatically.
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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# ADAPTIVE CSS  (follows OS/browser preference via prefers-color-scheme)
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── CSS custom properties — light default ────────────────────── */
:root {
    --bg:        #F8FAFC;
    --surface:   #FFFFFF;
    --border:    #E2E8F0;
    --text:      #0F172A;
    --text-muted:#64748B;
    --blue:      #2563EB;
    --violet:    #7C3AED;
    --green:     #16A34A;
    --amber:     #D97706;
    --red:       #DC2626;
    --chip-blue-bg:  #DBEAFE;
    --chip-blue-fg:  #1D4ED8;
    --chip-green-bg: #DCFCE7;
    --chip-green-fg: #15803D;
    --chip-red-bg:   #FEE2E2;
    --chip-red-fg:   #B91C1C;
    --chip-muted-bg: #F1F5F9;
    --chip-muted-fg: #64748B;
    --plot-bg:   #F1F5F9;
    --hero-text: #FFFFFF;
}

/* ── Dark mode overrides ──────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
    :root {
        --bg:        #0F172A;
        --surface:   #1E293B;
        --border:    #334155;
        --text:      #F1F5F9;
        --text-muted:#94A3B8;
        --chip-blue-bg:  #1E3A5F;
        --chip-blue-fg:  #60A5FA;
        --chip-green-bg: #14532D;
        --chip-green-fg: #4ADE80;
        --chip-red-bg:   #450A0A;
        --chip-red-fg:   #F87171;
        --chip-muted-bg: #1E293B;
        --chip-muted-fg: #94A3B8;
        --plot-bg:   #1E293B;
    }
}

/* ── Streamlit dark-mode class override ────────────────────────── */
[data-theme="dark"] {
    --bg:        #0F172A;
    --surface:   #1E293B;
    --border:    #334155;
    --text:      #F1F5F9;
    --text-muted:#94A3B8;
    --chip-blue-bg:  #1E3A5F;
    --chip-blue-fg:  #60A5FA;
    --chip-green-bg: #14532D;
    --chip-green-fg: #4ADE80;
    --chip-red-bg:   #450A0A;
    --chip-red-fg:   #F87171;
    --chip-muted-bg: #1E293B;
    --chip-muted-fg: #94A3B8;
    --plot-bg:   #1E293B;
}

/* ── Base styles ─────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stSidebar"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: Inter, system-ui, -apple-system, sans-serif;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Metric cards ────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
}
[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: .03em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Typography ──────────────────────────────────────────────── */
h1 { font-size: 1.7rem  !important; font-weight: 800 !important;
     color: var(--text) !important; margin-bottom: 0 !important; }
h2 { font-size: 1.2rem  !important; font-weight: 700 !important;
     color: var(--text) !important; }
h3 { font-size: 1.0rem  !important; font-weight: 600 !important;
     color: var(--text) !important; }
p, li, span, div { color: var(--text); }

/* ── Divider ─────────────────────────────────────────────────── */
hr { border: none; border-top: 1px solid var(--border); margin: 18px 0; }

/* ── Tabs ─────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: var(--border) !important;
    border-radius: 8px;
    gap: 3px;
    padding: 3px;
}
[data-baseweb="tab"] {
    border-radius: 6px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    background: transparent !important;
}
[aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--blue) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.12) !important;
}

/* ── Cards ────────────────────────────────────────────────────── */
.qlop-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}

/* ── Hero banner ─────────────────────────────────────────────── */
.qlop-hero {
    background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
    border-radius: 14px;
    padding: 30px 34px;
    margin-bottom: 24px;
}
.qlop-hero h1 {
    color: #FFFFFF !important;
    font-size: 1.9rem !important;
    margin: 0 !important;
}
.qlop-hero p  { color: rgba(255,255,255,.85); margin: 8px 0 0; font-size: 15px; }

/* ── Pill chips ───────────────────────────────────────────────── */
.chip {
    display: inline-block;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px 2px;
    line-height: 1.6;
}
.chip-blue  { background: var(--chip-blue-bg);  color: var(--chip-blue-fg);  }
.chip-green { background: var(--chip-green-bg); color: var(--chip-green-fg); }
.chip-red   { background: var(--chip-red-bg);   color: var(--chip-red-fg);   }
.chip-muted { background: var(--chip-muted-bg); color: var(--chip-muted-fg); }

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {
    background: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 9px 22px;
    font-weight: 600;
    font-size: 13px;
    transition: opacity .15s;
}
.stButton > button:hover { opacity: .88; }

/* ── Select / multiselect ────────────────────────────────────── */
[data-baseweb="select"] > div,
[data-baseweb="multi-select"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ── Progress bar ────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: #2563EB !important;
}

/* ── Info / success / warning boxes ─────────────────────────── */
.stAlert { border-radius: 8px !important; }

/* ── Caption ─────────────────────────────────────────────────── */
.stCaption { color: var(--text-muted) !important; }

/* ── Plotly chart containers ─────────────────────────────────── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PLOTLY THEME  — transparent bg so it inherits the CSS surface
# ─────────────────────────────────────────────────────────────────
def make_layout(height=420, **kw):
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        margin=dict(t=36, b=20, l=10, r=10),
        xaxis=dict(gridcolor="rgba(128,128,128,.2)", zeroline=False),
        yaxis=dict(gridcolor="rgba(128,128,128,.2)", zeroline=False),
        **kw,
    )


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
C_BLUE   = "#2563EB"
C_VIOLET = "#7C3AED"
C_GREEN  = "#16A34A"
C_AMBER  = "#D97706"
C_RED    = "#DC2626"
C_MUTED  = "#94A3B8"

JOBS_PATH     = "data/processed/MASTERED_DATA_FINAL_MODELING.csv"
COURSERA_PATH = "data/raw/Coursera (1).csv"

# ─────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_jobs():
    df = pd.read_csv(JOBS_PATH)
    df["skills_list"] = df["hard_skills"].apply(
        lambda x: [s.strip() for s in str(x).split(",") if s.strip()]
        if pd.notna(x) else []
    )
    df["postedAt"] = pd.to_datetime(df["postedAt"], dayfirst=True, errors="coerce")
    df["year"]     = df["postedAt"].dt.year
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
        "Job category": "role_category",
        "Difficulty": "difficulty", "Duration": "duration",
    }, inplace=True)
    return df


@st.cache_data(ttl=3600)
def build_counters(df_jobs, df_course):
    """
    j_ctr  — LinkedIn skill frequencies (vocabulary reference)
    c_ctr  — Coursera skill frequencies, but ONLY for skills in LinkedIn vocab
    """
    j_all: list = []
    df_jobs["skills_list"].apply(lambda l: j_all.extend(l))
    j_ctr = Counter(j_all)

    linkedin_vocab = set(j_ctr.keys())
    c_matched: list = []
    for skills in df_course["skills_list"]:
        for sk in skills:
            if sk in linkedin_vocab:
                c_matched.append(sk)
    c_ctr = Counter(c_matched)
    return j_ctr, c_ctr, linkedin_vocab


@st.cache_data(ttl=3600)
def build_role_profiles(df):
    s_all: list = []
    df["skills_list"].apply(lambda l: s_all.extend(l))
    freq = Counter(s_all)
    sel  = sorted([s for s, c in freq.items() if c >= 15])
    mlb  = MultiLabelBinarizer(classes=sel)
    X    = mlb.fit_transform(df["skills_list"])
    Xdf  = pd.DataFrame(X, columns=mlb.classes_)
    Xdf["role_label"] = df["role_label"].values
    return Xdf.groupby("role_label").mean(), mlb, sel


# ─────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────
df    = load_jobs()
df_c  = load_coursera()
j_ctr, c_ctr, linkedin_vocab = build_counters(df, df_c)
role_profiles, mlb, sel_skills = build_role_profiles(df)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:14px 0 18px;">
        <div style="font-size:32px;margin-bottom:6px;">📊</div>
        <div style="font-size:16px;font-weight:800;">QLOP</div>
        <div style="font-size:12px;color:var(--text-muted);margin-top:2px;">
            Market Insight Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "nav",
        ["Overview", "Skill Demand", "Hiring Trends", "Course Supply", "A/B Testing"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Loker: {len(df):,} · {df['role_label'].nunique()} roles")
    st.caption(f"Course: {len(df_c):,} · Skill LinkedIn: {len(j_ctr):,}")
    st.caption("LinkedIn Indonesia 2022–2026 · Coursera")

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
SEN_ORDER = ["Entry level","Associate","Mid-Senior level","Director","Executive"]


def bar_h(df_in, x, y, color_scale="Blues", text_col=None, height=400, title=""):
    fig = px.bar(
        df_in, x=x, y=y, orientation="h",
        text=text_col or x,
        color=x, color_continuous_scale=color_scale,
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title=x, yaxis_title="",
        title=title, title_font_size=14,
    )
    fig.update_layout(**make_layout(height))
    return fig


# ─────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown("""
    <div class="qlop-hero">
      <h1>QLOP — Market Insight Dashboard</h1>
      <p>Wawasan makro pasar kerja IT Indonesia untuk mendukung sistem analisis skills gap QLOP</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Loker",        f"{len(df):,}")
    k2.metric("IT Roles",           df["role_label"].nunique())
    k3.metric("Skill Unik",         f"{len(j_ctr):,}")
    k4.metric("Total Course",       f"{len(df_c):,}")
    remote_pct = df["workRemoteAllowed"].fillna(0).astype(float).mean()*100
    k5.metric("Remote Work",        f"{remote_pct:.1f}%")
    k6.metric("Rata-rata Skill/Loker", f"{df['skill_count'].mean():.1f}")

    st.divider()

    # Row 1: role distribution + yearly trend
    col_a, col_b = st.columns([3, 2], gap="medium")
    with col_a:
        st.subheader("Distribusi Loker per Role IT")
        rc = df["role_label"].value_counts().reset_index()
        rc.columns = ["role", "count"]
        fig = bar_h(rc, "count", "role", height=620, title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Tren Posting Loker per Tahun")
        yt = df[df["year"].between(2022,2026)].groupby("year").size().reset_index(name="count")
        fig2 = px.area(yt, x="year", y="count", markers=True,
                       color_discrete_sequence=[C_BLUE])
        fig2.update_traces(fill="tozeroy", fillcolor="rgba(37,99,235,.12)",
                           line_width=2.5, marker_size=7)
        fig2.update_layout(xaxis_title="Tahun", yaxis_title="Loker",
                           **make_layout(235))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Employment Type")
        em = df["employmentType"].value_counts().reset_index()
        em.columns = ["type","count"]
        fig3 = px.pie(em, values="count", names="type", hole=0.55,
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig3.update_traces(textposition="inside", textinfo="percent+label",
                           textfont_size=11)
        fig3.update_layout(showlegend=False, **make_layout(270))
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # Row 2: top skills
    st.subheader("Top 10 Skill — Demand (Loker) vs Supply (Coursera)")
    ca, cb = st.columns(2, gap="medium")
    with ca:
        t10j = pd.DataFrame(j_ctr.most_common(10), columns=["skill","count"])
        fig = px.bar(t10j, x="skill", y="count", text="count",
                     color_discrete_sequence=[C_BLUE])
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(xaxis_tickangle=-35, xaxis_title="", yaxis_title="Frekuensi",
                          title="Skill Paling Dicari (LinkedIn)", title_font_size=14,
                          **make_layout(320))
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        # c_ctr: Coursera skills that match LinkedIn vocab
        t10c = pd.DataFrame(c_ctr.most_common(10), columns=["skill","count"])
        fig = px.bar(t10c, x="skill", y="count", text="count",
                     color_discrete_sequence=[C_VIOLET])
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(xaxis_tickangle=-35, xaxis_title="", yaxis_title="Frekuensi",
                          title="Skill Tersedia di Coursera (sesuai LinkedIn vocab)",
                          title_font_size=14, **make_layout(320))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Row 3: seniority + remote snapshot
    cs, cr = st.columns(2, gap="medium")
    with cs:
        st.subheader("Distribusi Seniority Level")
        df_sen = df[df["seniorityLevel"].isin(SEN_ORDER)]
        sc = (df_sen["seniorityLevel"].value_counts()
              .reindex(SEN_ORDER).dropna().reset_index())
        sc.columns = ["level","count"]
        fig = px.bar(sc, x="level", y="count", text="count",
                     color="level",
                     color_discrete_sequence=px.colors.sequential.Blues_r[:len(sc)])
        fig.update_traces(textposition="outside", marker_line_width=0, showlegend=False)
        fig.update_layout(xaxis_title="", yaxis_title="Loker", **make_layout(260))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.subheader("Remote Work — Top 10 Role")
        df_rm = df.dropna(subset=["workRemoteAllowed"]).copy()
        df_rm["is_remote"] = df_rm["workRemoteAllowed"].astype(float).astype(bool)
        rr = (df_rm.groupby("role_label")
              .agg(total=("is_remote","count"), remote=("is_remote","sum"))
              .assign(pct=lambda x: (x.remote/x.total*100).round(1))
              .nlargest(10,"pct").reset_index())
        avg_r = df_rm["is_remote"].mean() * 100
        fig = px.bar(rr, x="pct", y="role_label", orientation="h", text="pct",
                     color="pct", color_continuous_scale="Greens")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        fig.add_vline(x=avg_r, line_dash="dash", line_color=C_AMBER,
                      annotation_text=f"Avg {avg_r:.1f}%",
                      annotation_position="top right")
        fig.update_layout(yaxis={"categoryorder":"total ascending"},
                          xaxis_title="% Remote", yaxis_title="",
                          **make_layout(260))
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: SKILL DEMAND
# ─────────────────────────────────────────────────────────────────
elif page == "Skill Demand":
    st.title("Skill Demand — Pasar Kerja IT")
    st.caption("Skill teknis yang paling banyak dicari berdasarkan lowongan kerja LinkedIn Indonesia")

    tab1, tab2, tab3 = st.tabs(["Top Skills Global", "Skills per Role", "Heatmap"])

    with tab1:
        r1, r2 = st.columns([1, 3], gap="medium")
        with r1:
            top_n  = st.slider("Top N Skills", 10, 50, 25)
            yr_sel = st.multiselect("Tahun", [2022,2023,2024,2025,2026],
                                    default=[2022,2023,2024,2025,2026])
        dff = df[df["year"].isin(yr_sel)] if yr_sel else df
        s_all: list = []
        dff["skills_list"].apply(lambda l: s_all.extend(l))
        ctr_f = Counter(s_all)
        tsk   = pd.DataFrame(ctr_f.most_common(top_n), columns=["skill","count"])
        with r2:
            fig = bar_h(tsk, "count", "skill", height=max(420, top_n*22))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tren Top 8 Skill per Tahun")
        top8 = [s for s,_ in ctr_f.most_common(8)]
        rows = []
        for _, row in df.dropna(subset=["year"]).iterrows():
            yr = int(row["year"])
            if 2022 <= yr <= 2026:
                for sk in row["skills_list"]:
                    if sk in top8:
                        rows.append({"Year": yr, "Skill": sk})
        if rows:
            tgrp = pd.DataFrame(rows).groupby(["Year","Skill"]).size().reset_index(name="Count")
            fig2 = px.line(tgrp, x="Year", y="Count", color="Skill", markers=True,
                           line_shape="spline",
                           color_discrete_sequence=px.colors.qualitative.Plotly)
            fig2.update_traces(line_width=2.5)
            fig2.update_layout(**make_layout(380))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        rs  = st.selectbox("Pilih Role IT", sorted(df["role_label"].unique()))
        ns  = st.slider("Jumlah Skill", 5, 20, 12)
        dfr = df[df["role_label"] == rs]
        rs_all: list = []
        dfr["skills_list"].apply(lambda l: rs_all.extend(l))
        rctr  = Counter(rs_all)
        tprol = pd.DataFrame(rctr.most_common(ns), columns=["skill","count"])
        c1, c2 = st.columns([2,1], gap="medium")
        with c1:
            if not tprol.empty:
                fig = bar_h(tprol, "count", "skill", height=max(320, ns*36),
                            title=f"Top {ns} Skill — {rs} ({len(dfr):,} loker)")
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
                st.markdown(f"<span class='chip chip-blue'>{s} &nbsp;{pct:.0f}%</span>",
                            unsafe_allow_html=True)

    with tab3:
        st.subheader("Heatmap: % Kemunculan Skill per Role")
        nh       = st.slider("Jumlah top skill", 10, 30, 20)
        th_skills = [s for s,_ in j_ctr.most_common(nh)]
        th_roles  = df["role_label"].value_counts().head(15).index.tolist()
        hm = {
            role: {
                sk: round(
                    df[df["role_label"]==role]["skills_list"]
                    .apply(lambda l: sk in l).sum()
                    / max(len(df[df["role_label"]==role]),1)*100, 1)
                for sk in th_skills
            }
            for role in th_roles
        }
        hm_df = pd.DataFrame(hm).T
        fig = px.imshow(hm_df, color_continuous_scale="Blues",
                        aspect="auto", text_auto=".0f")
        fig.update_layout(coloraxis_colorbar_title="%")
        fig.update_xaxes(tickangle=-40)
        fig.update_layout(**make_layout(520))
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: HIRING TRENDS
# ─────────────────────────────────────────────────────────────────
elif page == "Hiring Trends":
    st.title("Hiring Trends")
    st.caption("Tren perekrutan, kompetisi, seniority, dan remote work di IT Indonesia")

    tab1, tab2, tab3 = st.tabs(["Kompetisi", "Remote Work", "Seniority & Employment"])

    with tab1:
        st.subheader("Kompetisi per Role — Rata-rata Jumlah Pelamar")
        comp = (df.groupby("role_label")
                .agg(avg_applicants=("applicantsCount","mean"),
                     total_jobs=("role_label","count"))
                .round(1).sort_values("avg_applicants", ascending=False).reset_index())
        c1, c2 = st.columns([3,2], gap="medium")
        with c1:
            fig = px.bar(comp, x="avg_applicants", y="role_label", orientation="h",
                         text="avg_applicants", color="avg_applicants",
                         color_continuous_scale="RdYlGn_r")
            fig.update_traces(texttemplate="%{text:.0f}", textposition="outside",
                              marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            fig.update_layout(yaxis={"categoryorder":"total ascending"},
                              xaxis_title="Rata-rata Pelamar", yaxis_title="",
                              **make_layout(620))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.scatter(comp, x="total_jobs", y="avg_applicants",
                              size="total_jobs", color="avg_applicants",
                              color_continuous_scale="RdYlGn_r", text="role_label",
                              title="Loker vs Kompetisi")
            fig2.update_traces(textposition="top center", textfont_size=9)
            fig2.update_coloraxes(showscale=False)
            fig2.update_layout(xaxis_title="Total Loker", yaxis_title="Avg Pelamar",
                               **make_layout(620))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Adopsi Remote Work per Role IT")
        df_rm = df.dropna(subset=["workRemoteAllowed"]).copy()
        df_rm["is_remote"] = df_rm["workRemoteAllowed"].astype(float).astype(bool)
        rr2 = (df_rm.groupby("role_label")
               .agg(total=("is_remote","count"), remote=("is_remote","sum"))
               .assign(pct=lambda x: (x.remote/x.total*100).round(1))
               .sort_values("pct", ascending=False).reset_index())
        avg_r = df_rm["is_remote"].mean() * 100
        c1, c2 = st.columns([3,2], gap="medium")
        with c1:
            fig = px.bar(rr2, x="pct", y="role_label", orientation="h", text="pct",
                         color="pct", color_continuous_scale="Greens")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            fig.add_vline(x=avg_r, line_dash="dash", line_color=C_AMBER,
                          annotation_text=f"Avg {avg_r:.1f}%")
            fig.update_layout(yaxis={"categoryorder":"total ascending"},
                              xaxis_title="% Remote", yaxis_title="",
                              **make_layout(560))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            yes = df_rm["is_remote"].sum(); no = (~df_rm["is_remote"]).sum()
            fig2 = px.pie(values=[yes,no], names=["Remote","On-site"], hole=0.55,
                          color_discrete_sequence=[C_GREEN,"#CBD5E1"])
            fig2.update_traces(textposition="inside", textinfo="percent+label",
                               textfont_size=12)
            fig2.update_layout(showlegend=False, **make_layout(270))
            st.plotly_chart(fig2, use_container_width=True)
            st.metric("Rata-rata Remote",       f"{avg_r:.1f}%")
            st.metric("Paling Remote-friendly", rr2.iloc[0]["role_label"])

    with tab3:
        st.subheader("Distribusi Seniority Level")
        df_sen = df[df["seniorityLevel"].isin(SEN_ORDER)].copy()
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            sc = df_sen["seniorityLevel"].value_counts().reset_index()
            sc.columns = ["level","count"]
            fig = px.pie(sc, values="count", names="level", hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              textfont_size=11)
            fig.update_layout(**make_layout(300))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sr  = df_sen.groupby(["role_label","seniorityLevel"]).size().unstack(fill_value=0)
            sp  = sr.div(sr.sum(axis=1),axis=0)*100
            top8 = df["role_label"].value_counts().head(8).index
            sp   = sp.reindex(top8)
            ex   = [c for c in SEN_ORDER if c in sp.columns]
            fig2 = px.bar(sp[ex].reset_index(), y="role_label",
                          x=ex, barmode="stack", orientation="h",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(xaxis_title="%", yaxis_title="", legend_title="Level",
                               **make_layout(300))
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
            fig.update_layout(showlegend=False, **make_layout(270))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            top6 = df["role_label"].value_counts().head(6).index
            er = (df[df["role_label"].isin(top6)]
                  .groupby(["role_label","employmentType"]).size().unstack(fill_value=0))
            ep = er.div(er.sum(axis=1),axis=0)*100
            fig2 = px.bar(ep.reset_index(), y="role_label",
                          x=[c for c in ep.columns], barmode="stack", orientation="h",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(xaxis_title="%", yaxis_title="", legend_title="Tipe",
                               **make_layout(270))
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: COURSE SUPPLY
# ─────────────────────────────────────────────────────────────────
elif page == "Course Supply":
    st.title("Course Supply — Analisis Coursera")
    st.caption(
        "Ketersediaan course untuk menutup kesenjangan skill di pasar IT Indonesia. "
        "Supply dihitung hanya untuk skill yang ada dalam kosakata LinkedIn."
    )

    tab1, tab2, tab3 = st.tabs(["Overview Course", "Supply vs Demand", "Course per Role"])

    with tab1:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Course",     f"{len(df_c):,}")
        k2.metric("Role Tercakup",    df_c["role_category"].nunique())
        k3.metric("Skill LinkedIn\ntercover di Coursera", f"{len(c_ctr)}")
        k4.metric("Avg Skill/Course", f"{df_c['skill_count'].mean():.1f}")

        st.divider()
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.subheader("Distribusi Course per Role (Top 15)")
            cc = df_c["role_category"].value_counts().head(15).reset_index()
            cc.columns = ["role","count"]
            fig = bar_h(cc, "count", "role", color_scale="Purples", height=440)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Tingkat Kesulitan")
            di = df_c["difficulty"].value_counts().reset_index()
            di.columns = ["level","count"]
            fig2 = px.pie(di, values="count", names="level", hole=0.55,
                          color_discrete_sequence=[C_GREEN,C_BLUE,C_AMBER,C_RED])
            fig2.update_traces(textposition="inside", textinfo="percent+label",
                               textfont_size=12)
            fig2.update_layout(**make_layout(200))
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Durasi Course")
            du = df_c["duration"].value_counts().reset_index()
            du.columns = ["durasi","count"]
            du["durasi"] = du["durasi"].str.replace("_"," ").str.title()
            fig3 = px.bar(du, x="durasi", y="count", text="count",
                          color="count", color_continuous_scale="Blues")
            fig3.update_traces(textposition="outside", marker_line_width=0)
            fig3.update_coloraxes(showscale=False)
            fig3.update_layout(xaxis_tickangle=-30, xaxis_title="", yaxis_title="Jumlah",
                               **make_layout(200))
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.subheader("Skill Supply vs Demand")
        st.info(
            "Supply dihitung berdasarkan **berapa banyak course Coursera** yang mengajarkan "
            "skill tertentu dari kosakata LinkedIn. Skill dengan demand tinggi tapi supply "
            "rendah adalah **prioritas pembelajaran** yang perlu diperhatikan."
        )
        TOP_N = st.slider("Jumlah skill top demand", 15, 50, 30)

        # Build gap data: only LinkedIn vocab
        top_d = dict(j_ctr.most_common(TOP_N))
        gdf = pd.DataFrame([
            {
                "skill":     sk,
                "demand":    dm,
                "supply":    c_ctr.get(sk, 0),
                "gap_ratio": dm / max(c_ctr.get(sk, 0), 1),
                "covered":   c_ctr.get(sk, 0) > 0,
            }
            for sk, dm in top_d.items()
        ])
        gdf_sorted = gdf.sort_values("demand", ascending=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            # Grouped bar demand vs supply
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Demand (Loker)", y=gdf_sorted["skill"], x=gdf_sorted["demand"],
                orientation="h", marker_color=C_BLUE, marker_line_width=0,
            ))
            fig.add_trace(go.Bar(
                name="Supply (Coursera, matched)", y=gdf_sorted["skill"], x=gdf_sorted["supply"],
                orientation="h", marker_color=C_VIOLET, marker_line_width=0,
            ))
            fig.update_layout(
                barmode="group",
                yaxis={"categoryorder":"array",
                       "categoryarray": gdf_sorted["skill"].tolist()},
                xaxis_title="Frekuensi", yaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                **make_layout(max(500, TOP_N*25)),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Skill Belum Terlayani (Top 15 Gap)")
            st.caption(
                "Skill dengan demand tinggi namun belum ada course Coursera "
                "yang secara eksplisit mengajarkannya."
            )
            not_covered = gdf[gdf["supply"] == 0].sort_values("demand", ascending=False).head(15)
            fig2 = px.bar(
                not_covered, x="demand", y="skill", orientation="h",
                text="demand", color="demand", color_continuous_scale="Reds",
            )
            fig2.update_traces(textposition="outside", marker_line_width=0)
            fig2.update_coloraxes(showscale=False)
            fig2.update_layout(
                yaxis={"categoryorder":"total ascending"},
                xaxis_title="Demand (Frekuensi Loker)", yaxis_title="",
                **make_layout(max(380, 15*28)),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Categorisation  — only 2 groups: covered & not covered
        st.divider()
        st.subheader("Kategorisasi Skill (berdasarkan kosakata LinkedIn)")
        covered_df     = gdf[gdf["covered"]].sort_values("demand", ascending=False)
        not_covered_df = gdf[~gdf["covered"]].sort_values("demand", ascending=False)

        ca, cb = st.columns(2, gap="medium")
        with ca:
            n_cov = len(covered_df)
            st.markdown(f"**Tercakup di Coursera** &nbsp; ({n_cov} dari {TOP_N} skill)")
            for _, row in covered_df.iterrows():
                st.markdown(
                    f"<span class='chip chip-green'>{row['skill']}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
        with cb:
            n_not = len(not_covered_df)
            st.markdown(f"**Belum Tercakup** &nbsp; ({n_not} dari {TOP_N} skill)")
            for _, row in not_covered_df.iterrows():
                st.markdown(
                    f"<span class='chip chip-red'>{row['skill']}"
                    f"</span>",
                    unsafe_allow_html=True,
                )

    with tab3:
        st.subheader("Course Tersedia per Role IT")
        rs3   = st.selectbox("Pilih Role", sorted(df["role_label"].unique()), key="cs_role")
        df_cr = df_c[df_c["role_category"] == rs3]
        if df_cr.empty:
            df_cr = df_c[df_c["role_category"].str.contains(
                rs3.split()[0], case=False, na=False)]

        c1, c2 = st.columns([2,1], gap="medium")
        with c1:
            if not df_cr.empty:
                st.success(f"Ditemukan **{len(df_cr)} course** untuk role **{rs3}**")
                for _, row in df_cr.head(15).iterrows():
                    diff_col = {
                        "BEGINNER": C_GREEN, "INTERMEDIATE": C_AMBER,
                        "ADVANCED": C_RED,   "MIXED": C_BLUE,
                    }.get(str(row.get("difficulty","")), C_MUTED)
                    dur_str  = str(row.get("duration","")).replace("_"," ").title()
                    skp = row["skills_list"][:6]
                    more = f" +{len(row['skills_list'])-6} lainnya" \
                           if len(row["skills_list"]) > 6 else ""
                    st.markdown(f"""
                    <div class="qlop-card">
                        <div style="font-size:14px;font-weight:700;">
                            <a href="{row['url']}" target="_blank"
                               style="color:{C_BLUE};text-decoration:none;">
                                {row['name']}
                            </a>
                        </div>
                        <div style="font-size:12px;color:var(--text-muted);margin:4px 0;">
                            {row['partner']} &nbsp;·&nbsp;
                            <span style="color:{diff_col};font-weight:600;">
                                {row.get('difficulty','')}
                            </span>
                            &nbsp;·&nbsp; {dur_str}
                        </div>
                        <div style="font-size:12px;">
                            {", ".join(skp)}{more}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"Belum ada course yang ter-mapping ke role '{rs3}'.")

        with c2:
            # Skill coverage: LinkedIn vocab skills demanded by this role
            j_rs: list = []
            df[df["role_label"]==rs3]["skills_list"].apply(lambda l: j_rs.extend(l))
            jrc   = Counter(j_rs)
            top_j = set(s for s,_ in jrc.most_common(10))

            c_rs: list = []
            df_cr["skills_list"].apply(lambda l: c_rs.extend(l))
            crc   = Counter(c_rs)
            top_c = set(s for s,_ in crc.most_common(50)) & linkedin_vocab

            cov = top_j & top_c
            unc = top_j - top_c
            cov_pct = len(cov)/max(len(top_j),1)*100
            st.markdown(f"**Skill Coverage — {rs3}**")
            st.metric("Skill terlayani (top 10)", f"{len(cov)} / {len(top_j)}")
            st.progress(cov_pct/100, text=f"Coverage {cov_pct:.0f}%")
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
    st.title("A/B Testing — Strategi Rekomendasi Role")
    st.caption(
        "Eksperimen validasi: Top-3 Heuristik vs Cosine Similarity "
        "untuk mapping skill CV ke role IT (konteks AI Engine QLOP)"
    )
    st.info(
        "**H₀:** Tidak ada perbedaan signifikan &nbsp;·&nbsp; "
        "**H₁:** Strategi B lebih akurat &nbsp;·&nbsp; "
        "**α = 0.05** &nbsp;·&nbsp; Metrik: Precision@1 &nbsp;·&nbsp; N = 20% test set"
    )

    @st.cache_data(ttl=7200)
    def run_ab():
        from sklearn.model_selection import train_test_split
        from scipy import stats as sp

        dfab = pd.read_csv(JOBS_PATH)
        dfab["skills_list"] = dfab["hard_skills"].apply(
            lambda x: [s.strip() for s in str(x).split(",") if s.strip()]
            if pd.notna(x) else []
        )
        s_all: list = []
        dfab["skills_list"].apply(lambda l: s_all.extend(l))
        freq = Counter(s_all)
        sel2 = sorted([s for s,c in freq.items() if c >= 20])

        mlb2  = MultiLabelBinarizer(classes=sel2)
        mlb2.fit_transform(dfab["skills_list"])

        tr, te = train_test_split(dfab, test_size=0.2, random_state=42,
                                  stratify=dfab["role_label"])
        Xtr = mlb2.transform(tr["skills_list"])
        Xte = mlb2.transform(te["skills_list"])
        yte = te["role_label"].values

        trm = pd.DataFrame(Xtr, columns=mlb2.classes_)
        trm["role_label"] = tr["role_label"].values
        rp = trm.groupby("role_label").mean()
        ro = rp.index.tolist()

        rt3 = {r: set(rp.loc[r].nlargest(3).index) for r in ro}
        def pred_a(skills):
            best, bs = None, -1
            for r, t3 in rt3.items():
                sc = len(set(skills) & t3)
                if sc > bs: bs, best = sc, r
            return best or "Software Engineer"

        pA = [pred_a(te["skills_list"].iloc[i]) for i in range(len(te))]
        pB = [ro[np.argmax(cosine_similarity(Xte[i:i+1], rp.values)[0])]
              for i in range(len(Xte))]

        aA = np.mean(np.array(pA) == yte)
        aB = np.mean(np.array(pB) == yte)
        cA = (np.array(pA) == yte).astype(int)
        cB = (np.array(pB) == yte).astype(int)
        b  = ((cA==1)&(cB==0)).sum(); c = ((cA==0)&(cB==1)).sum()
        st2 = (abs(b-c)-1)**2/(b+c) if (b+c)>0 else 0
        pv  = sp.chi2.sf(st2, df=1)
        np.random.seed(42)
        diffs = np.array([
            cB[np.random.choice(len(cA),len(cA),replace=True)].mean() -
            cA[np.random.choice(len(cA),len(cA),replace=True)].mean()
            for _ in range(3000)
        ])
        return aA, aB, pv, diffs, len(yte)

    with st.spinner("Menjalankan eksperimen A/B..."):
        aA, aB, pv, boot, n_te = run_ab()

    ci_lo  = np.percentile(boot, 2.5)*100
    ci_hi  = np.percentile(boot, 97.5)*100
    mean_d = boot.mean()*100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Strategi A", f"{aA*100:.2f}%", help="Top-3 Heuristik")
    k2.metric("Strategi B", f"{aB*100:.2f}%",
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
        fig.update_layout(yaxis_range=[0, max(aA,aB)*130],
                          yaxis_title="Precision@1 (%)",
                          **make_layout(330))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Bootstrap Distribution (Selisih B − A)")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=boot*100, nbinsx=50,
                                    marker_color=C_BLUE, opacity=0.75))
        fig2.add_vline(x=mean_d, line_dash="dash",  line_color=C_RED,
                       annotation_text=f"Mean {mean_d:+.2f}%")
        fig2.add_vline(x=ci_lo,  line_dash="dot",   line_color=C_AMBER,
                       annotation_text=f"{ci_lo:+.1f}%")
        fig2.add_vline(x=ci_hi,  line_dash="dot",   line_color=C_AMBER,
                       annotation_text=f"{ci_hi:+.1f}%")
        fig2.add_vline(x=0, line_color=C_MUTED, opacity=0.5)
        fig2.update_layout(xaxis_title="Selisih Akurasi B−A (%)",
                           yaxis_title="Frekuensi",
                           **make_layout(330))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"95% CI: [{ci_lo:+.2f}%, {ci_hi:+.2f}%] — {n_te:,} sampel test")

    st.divider()
    st.subheader("Kesimpulan & Rekomendasi untuk QLOP")
    if pv < 0.05:
        winner = "Strategi B (Cosine Similarity)" if aB > aA else "Strategi A (Top-3)"
        st.success(
            f"**H₀ ditolak** (p = {pv:.4f} < 0.05)  \n"
            f"Terdapat perbedaan **signifikan** secara statistik.  \n"
            f"**{winner}** terbukti lebih akurat.  \n\n"
            f"Selisih: {(aB-aA)*100:+.2f}% (95% CI: [{ci_lo:+.2f}%, {ci_hi:+.2f}%])"
        )
    else:
        st.warning(
            f"**Gagal tolak H₀** (p = {pv:.4f} ≥ 0.05)  \n"
            f"Tidak ada perbedaan signifikan secara statistik.  \n\n"
            f"**Rekomendasi untuk QLOP:** Gunakan **Strategi B (Cosine Similarity)** — "
            f"lebih scalable dan memanfaatkan seluruh dimensi skill vektor."
        )
