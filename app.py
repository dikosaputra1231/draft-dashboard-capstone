"""
app.py - QLOP Market Insight Dashboard
=======================================
Dashboard insight pasar kerja IT Indonesia sebagai fitur pendukung
aplikasi QLOP (NLP-driven Skill Gap Analysis Tool).

Peran: Data Science - menyediakan wawasan makro pasar kerja IT.

Halaman:
  1. Overview          - ringkasan dataset loker & course, KPI
  2. Skill Demand      - skill paling dicari per role, tren tahunan, heatmap
  3. Hiring Trends     - tren posting loker, kompetisi, seniority, remote work
  4. Course Supply     - analisis Coursera, skill coverage, gap supply vs demand
  5. A/B Testing       - hasil eksperimen strategi rekomendasi

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

warnings.filterwarnings('ignore')

# ===========================================================
# KONFIGURASI HALAMAN
# ===========================================================
st.set_page_config(
    page_title="QLOP - Market Insight Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================
# CSS CUSTOM - Dark theme premium
# ===========================================================
st.markdown("""
<style>
    .stApp { background-color: #0F172A; color: #E2E8F0; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        border-right: 1px solid #334155;
    }
    [data-testid="metric-container"] {
        background: #1E293B; border: 1px solid #334155;
        border-radius: 12px; padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] label { color: #94A3B8 !important; font-size: 13px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #F1F5F9 !important; font-size: 28px; font-weight: 700;
    }
    h1, h2, h3 { color: #F1F5F9 !important; }
    h1 { font-size: 2rem !important; font-weight: 800 !important; }
    .gradient-header {
        background: linear-gradient(135deg, #1D4ED8 0%, #7C3AED 100%);
        border-radius: 16px; padding: 24px; margin-bottom: 24px; text-align: center;
    }
    .gradient-header h1 { color: white !important; margin: 0; }
    .gradient-header p { color: #BFDBFE; margin: 8px 0 0; font-size: 16px; }
    .insight-card {
        background: #1E293B; border: 1px solid #334155;
        border-radius: 12px; padding: 20px; margin: 8px 0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    .skill-tag {
        display: inline-block; background: #1D4ED8; color: #DBEAFE;
        border-radius: 20px; padding: 4px 12px; margin: 3px;
        font-size: 13px; font-weight: 500;
    }
    .skill-tag-gap { background: #B91C1C; color: #FEE2E2; }
    .skill-tag-ok  { background: #15803D; color: #DCFCE7; }
    hr { border-color: #334155; }
    .stButton > button {
        background: linear-gradient(135deg, #1D4ED8, #7C3AED);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] { background: #1E293B; border-radius: 8px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { color: #94A3B8; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background: #1D4ED8 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ===========================================================
# PLOTLY THEME
# ===========================================================
PLOTLY_THEME = {
    'paper_bgcolor': '#0F172A', 'plot_bgcolor': '#1E293B',
    'font': {'color': '#E2E8F0', 'family': 'Inter, sans-serif'},
    'xaxis': {'gridcolor': '#334155', 'color': '#94A3B8'},
    'yaxis': {'gridcolor': '#334155', 'color': '#94A3B8'},
}

# ===========================================================
# DATA LOADING
# ===========================================================
@st.cache_data(ttl=3600)
def load_jobs():
    df = pd.read_csv("data/processed/MASTERED_DATA_FINAL_MODELING.csv")
    df['skills_list'] = df['hard_skills'].apply(
        lambda x: [s.strip() for s in str(x).split(',') if s.strip()] if pd.notna(x) else []
    )
    df['postedAt'] = pd.to_datetime(df['postedAt'], dayfirst=True, errors='coerce')
    df['year'] = df['postedAt'].dt.year
    return df

@st.cache_data(ttl=3600)
def load_coursera():
    df = pd.read_csv("data/raw/coursera.csv", sep=';', encoding='latin-1')
    # Gabung 19 kolom skill jadi 1 list
    skill_cols = [c for c in df.columns if c.strip().startswith('Skills') or c.strip().startswith('Skills ')]
    # Kolom pertama bernama 'Skills ' (ada spasi)
    skill_cols = ['Skills '] + [f'Skills {i}' for i in range(2, 20)]
    skill_cols = [c for c in skill_cols if c in df.columns]

    def merge_skills(row):
        skills = []
        for c in skill_cols:
            v = row[c]
            if pd.notna(v) and str(v).strip():
                skills.append(str(v).strip())
        return skills

    df['skills_list'] = df.apply(merge_skills, axis=1)
    df['skill_count'] = df['skills_list'].apply(len)
    df.rename(columns={'Name': 'name', 'Patners 1': 'partner',
                        'Url': 'url', 'Job category': 'role_category',
                        'Difficulty': 'difficulty', 'Duration': 'duration'}, inplace=True)
    return df

@st.cache_data(ttl=3600)
def build_skill_counter(df):
    all_skills = []
    df['skills_list'].apply(lambda l: all_skills.extend(l))
    return Counter(all_skills)

# ===========================================================
# SIDEBAR
# ===========================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0;'>
        <div style='font-size: 36px;'>📊</div>
        <div style='font-size: 18px; font-weight: 700; color: #F1F5F9;'>QLOP</div>
        <div style='font-size: 13px; color: #64748B;'>Market Insight Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigasi",
        ["🏠 Overview", "🔍 Skill Demand", "📈 Hiring Trends",
         "🎓 Course Supply", "🧪 A/B Testing"],
        label_visibility="hidden"
    )
    st.divider()
    st.caption("Data Loker: LinkedIn Indonesia 2022-2026")
    st.caption("Data Course: Coursera (1.980 course)")
    st.caption("Capstone DBS x Dicoding 2026")

# ===========================================================
# LOAD DATA
# ===========================================================
df = load_jobs()
df_c = load_coursera()
job_skill_ctr = build_skill_counter(df)
course_skill_ctr = build_skill_counter(df_c)

# ===========================================================
# PAGE 1: OVERVIEW
# ===========================================================
if page == "🏠 Overview":
    st.markdown("""
    <div class='gradient-header'>
        <h1>📊 QLOP - Market Insight Dashboard</h1>
        <p>Wawasan makro pasar kerja IT Indonesia — Data LinkedIn & Coursera</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📋 Total Loker", f"{len(df):,}")
    c2.metric("🎭 IT Roles", df['role_label'].nunique())
    c3.metric("🛠️ Skill Unik (Loker)", f"{len(job_skill_ctr):,}")
    c4.metric("🎓 Total Course", f"{len(df_c):,}")
    c5.metric("🌐 Remote Work", f"{df['workRemoteAllowed'].fillna(0).astype(float).mean()*100:.1f}%")
    c6.metric("⚡ Avg Skill/Loker", f"{df['skill_count'].mean():.1f}")

    st.divider()
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("Distribusi Loker per Role IT")
        rc = df['role_label'].value_counts().reset_index()
        rc.columns = ['role', 'count']
        fig = px.bar(rc, x='count', y='role', orientation='h',
                     color='count', color_continuous_scale='viridis', text='count')
        fig.update_traces(textposition='outside', textfont_size=10)
        # KODE BARU (Benar)
        fig.update_layout(height=650, coloraxis_showscale=False, **PLOTLY_THEME)
        fig.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Tren Loker per Tahun")
        yearly = df[df['year'].between(2022, 2026)].groupby('year').size().reset_index(name='count')
        fig2 = px.area(yearly, x='year', y='count', markers=True,
                       color_discrete_sequence=['#2563EB'])
        fig2.update_traces(fill='tozeroy', fillcolor='rgba(37,99,235,0.2)',
                           line_width=3, marker_size=8)
        fig2.update_layout(height=260, xaxis_title='Tahun', yaxis_title='Jumlah Loker',
                           **PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Distribusi Course per Role")
        cc = df_c['role_category'].value_counts().head(10).reset_index()
        cc.columns = ['role', 'count']
        fig3 = px.bar(cc, x='count', y='role', orientation='h',
                      color='count', color_continuous_scale='Purples', text='count')
        fig3.update_traces(textposition='outside', textfont_size=10)
        fig3.update_layout(height=320, coloraxis_showscale=False, **PLOTLY_THEME)
        fig3.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig3, use_container_width=True)

    # Top skills side-by-side
    st.subheader("Top 10 Skill: Demand (Loker) vs Supply (Course)")
    col1, col2 = st.columns(2)
    with col1:
        top10_job = pd.DataFrame(job_skill_ctr.most_common(10), columns=['skill', 'count'])
        fig = px.bar(top10_job, x='skill', y='count', color='count',
                     color_continuous_scale='Blues', text='count',
                     title="Demand: Skill Paling Dicari di Loker")
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, coloraxis_showscale=False, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top10_course = pd.DataFrame(course_skill_ctr.most_common(10), columns=['skill', 'count'])
        fig = px.bar(top10_course, x='skill', y='count', color='count',
                     color_continuous_scale='Purples', text='count',
                     title="Supply: Skill Paling Banyak Diajarkan di Course")
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, coloraxis_showscale=False, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================
# PAGE 2: SKILL DEMAND
# ===========================================================
elif page == "🔍 Skill Demand":
    st.title("🔍 Skill Demand — Pasar Kerja IT")
    st.caption("Skill teknis yang paling banyak dicari berdasarkan lowongan pekerjaan")

    tab1, tab2, tab3 = st.tabs(["📊 Top Skills", "🎭 Skills per Role", "🔥 Heatmap"])

    with tab1:
        col1, col2 = st.columns([1, 3])
        with col1:
            top_n = st.slider("Top N Skills", 10, 50, 25)
            year_filter = st.multiselect("Filter Tahun", [2022, 2023, 2024, 2025, 2026],
                                          default=[2022, 2023, 2024, 2025, 2026])
        df_filt = df[df['year'].isin(year_filter)] if year_filter else df
        all_s = []
        df_filt['skills_list'].apply(lambda l: all_s.extend(l))
        ctr_f = Counter(all_s)
        top_skills = pd.DataFrame(ctr_f.most_common(top_n), columns=['skill', 'count'])

        with col2:
            fig = px.bar(top_skills, x='count', y='skill', orientation='h',
                         color='count', color_continuous_scale='viridis', text='count')
            fig.update_traces(textposition='outside')
            fig.update_layout(height=max(400, top_n * 22), coloraxis_showscale=False, **PLOTLY_THEME)
            fig.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tren Top 8 Skill per Tahun")
        top8 = [s for s, _ in ctr_f.most_common(8)]
        ts_rows = []
        for _, row in df.dropna(subset=['year']).iterrows():
            yr = int(row['year'])
            if not 2022 <= yr <= 2026:
                continue
            for sk in row['skills_list']:
                if sk in top8:
                    ts_rows.append({'Year': yr, 'Skill': sk})
        if ts_rows:
            ts_df = pd.DataFrame(ts_rows)
            ts_grp = ts_df.groupby(['Year', 'Skill']).size().reset_index(name='Count')
            fig2 = px.line(ts_grp, x='Year', y='Count', color='Skill',
                           markers=True, line_shape='spline',
                           color_discrete_sequence=px.colors.qualitative.Plotly)
            fig2.update_traces(line_width=2.5)
            fig2.update_layout(height=400, **PLOTLY_THEME)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        role_sel = st.selectbox("Pilih Role IT", sorted(df['role_label'].unique()))
        n_skills = st.slider("Jumlah Skill", 5, 20, 10)

        df_role = df[df['role_label'] == role_sel]
        role_s = []
        df_role['skills_list'].apply(lambda l: role_s.extend(l))
        role_ctr = Counter(role_s)
        top_role = pd.DataFrame(role_ctr.most_common(n_skills), columns=['skill', 'count'])

        col1, col2 = st.columns([2, 1])
        with col1:
            if not top_role.empty:
                fig = px.bar(top_role, x='count', y='skill', orientation='h',
                             color='count', color_continuous_scale='Blues', text='count')
                fig.update_traces(textposition='outside')
                fig.update_layout(height=max(300, n_skills * 35),
                                  yaxis={'categoryorder': 'total ascending'},
                                  coloraxis_showscale=False,
                                  title=f"Top {n_skills} Skill - {role_sel} ({len(df_role):,} loker)",
                                  **PLOTLY_THEME)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"**{role_sel}**")
            st.metric("Total Loker", f"{len(df_role):,}")
            st.metric("Avg Skill/Loker", f"{df_role['skill_count'].mean():.1f}")
            st.metric("Remote Work", f"{df_role['workRemoteAllowed'].fillna(0).astype(float).mean()*100:.1f}%")
            st.metric("Avg Pelamar", f"{df_role['applicantsCount'].mean():.0f}")
            st.markdown("**Skill Wajib:**")
            for skill, cnt in role_ctr.most_common(5):
                pct = cnt / len(df_role) * 100
                st.markdown(f"<span class='skill-tag'>{skill} ({pct:.0f}%)</span>",
                            unsafe_allow_html=True)

    with tab3:
        st.subheader("Heatmap: % Kemunculan Skill per Role")
        n_hm = st.slider("Jumlah top skill", 10, 30, 20)
        top_hm_skills = [s for s, _ in job_skill_ctr.most_common(n_hm)]
        top_hm_roles  = df['role_label'].value_counts().head(15).index.tolist()

        hm_data = {}
        for role in top_hm_roles:
            df_r = df[df['role_label'] == role]
            total = len(df_r)
            row_d = {}
            for sk in top_hm_skills:
                has = df_r['skills_list'].apply(lambda l: sk in l).sum()
                row_d[sk] = round(has / total * 100, 1)
            hm_data[role] = row_d

        hm_df = pd.DataFrame(hm_data).T
        fig = px.imshow(hm_df, color_continuous_scale='YlOrRd',
                        aspect='auto', text_auto='.0f')
        fig.update_layout(height=500, coloraxis_colorbar_title='%', **PLOTLY_THEME)
        fig.update_xaxes(tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================
# PAGE 3: HIRING TRENDS
# ===========================================================
elif page == "📈 Hiring Trends":
    st.title("📈 Hiring Trends")
    st.caption("Tren perekrutan, kompetisi, seniority, dan remote work di IT Indonesia")

    tab1, tab2, tab3 = st.tabs(["🏆 Kompetisi", "🌐 Remote Work", "📊 Seniority"])

    with tab1:
        st.subheader("Tingkat Kompetisi per Role (Rata-rata Pelamar)")
        comp = df.groupby('role_label').agg(
            avg_applicants=('applicantsCount', 'mean'),
            total_jobs=('role_label', 'count')
        ).round(1).sort_values('avg_applicants', ascending=False).reset_index()

        col1, col2 = st.columns([3, 2])
        with col1:
            fig = px.bar(comp, x='avg_applicants', y='role_label', orientation='h',
                         color='avg_applicants', color_continuous_scale='RdYlGn_r',
                         text='avg_applicants')
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(height=650, coloraxis_showscale=False,
                               yaxis={'categoryorder': 'total ascending'}, **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(comp, x='total_jobs', y='avg_applicants',
                              size='total_jobs', color='avg_applicants',
                              color_continuous_scale='RdYlGn_r', text='role_label')
            fig2.update_traces(textposition='top center', textfont_size=9)
            fig2.update_layout(height=650, coloraxis_showscale=False,
                                title="Loker vs Kompetisi", **PLOTLY_THEME)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Adopsi Remote Work per Role IT")
        df_rm = df.dropna(subset=['workRemoteAllowed']).copy()
        df_rm['is_remote'] = df_rm['workRemoteAllowed'].astype(float).astype(bool)

        remote_role = df_rm.groupby('role_label').agg(
            total=('is_remote', 'count'), remote_count=('is_remote', 'sum'),
        ).assign(pct=lambda x: (x['remote_count'] / x['total'] * 100).round(1)
        ).sort_values('pct', ascending=False).reset_index()

        col1, col2 = st.columns([3, 2])
        with col1:
            avg_remote = df_rm['is_remote'].mean() * 100
            fig = px.bar(remote_role, x='pct', y='role_label', orientation='h',
                         color='pct', color_continuous_scale='Greens', text='pct')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.add_vline(x=avg_remote, line_dash='dash', line_color='yellow',
                          annotation_text=f"Avg: {avg_remote:.1f}%")
            fig.update_layout(height=600, coloraxis_showscale=False,
                               yaxis={'categoryorder': 'total ascending'}, **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            overall_yes = df_rm['is_remote'].sum()
            overall_no  = (~df_rm['is_remote']).sum()
            fig2 = px.pie(values=[overall_yes, overall_no], names=['Remote', 'On-site'],
                          hole=0.55, color_discrete_sequence=['#16A34A', '#334155'])
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            fig2.update_layout(height=300, **PLOTLY_THEME)
            st.plotly_chart(fig2, use_container_width=True)
            st.metric("Rata-rata Remote", f"{avg_remote:.1f}%")
            st.metric("Paling Remote-friendly", remote_role.iloc[0]['role_label'])

    with tab3:
        st.subheader("Distribusi Jenjang Karir")
        SEN_ORDER = ['Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive']
        df_core_sen = df[df['seniorityLevel'].isin(SEN_ORDER)].copy()

        col1, col2 = st.columns(2)
        with col1:
            sen_counts = df_core_sen['seniorityLevel'].value_counts().reset_index()
            sen_counts.columns = ['level', 'count']
            fig = px.pie(sen_counts, values='count', names='level', hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350, **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sen_role = df_core_sen.groupby(['role_label', 'seniorityLevel']).size()\
                .unstack(fill_value=0)
            sen_pct = sen_role.div(sen_role.sum(axis=1), axis=0) * 100
            top8 = df['role_label'].value_counts().head(8).index
            sen_pct = sen_pct.reindex(top8)
            existing = [c for c in SEN_ORDER if c in sen_pct.columns]
            fig2 = px.bar(sen_pct[existing].reset_index(), y='role_label',
                          x=existing, barmode='stack', orientation='h',
                          labels={'value': '%', 'role_label': ''})
            fig2.update_layout(height=350, **PLOTLY_THEME)
            st.plotly_chart(fig2, use_container_width=True)

        # Employment type
        st.subheader("Employment Type")
        col1, col2 = st.columns(2)
        with col1:
            emp = df['employmentType'].value_counts().reset_index()
            emp.columns = ['type', 'count']
            fig = px.pie(emp, values='count', names='type', hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=300, showlegend=False, **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top6r = df['role_label'].value_counts().head(6).index
            emp_r = df[df['role_label'].isin(top6r)].groupby(['role_label','employmentType']).size().unstack(fill_value=0)
            emp_pct = emp_r.div(emp_r.sum(axis=1), axis=0) * 100
            fig2 = px.bar(emp_pct.reset_index(), y='role_label',
                          x=[c for c in emp_pct.columns], barmode='stack', orientation='h')
            fig2.update_layout(height=300, **PLOTLY_THEME)
            st.plotly_chart(fig2, use_container_width=True)

# ===========================================================
# PAGE 4: COURSE SUPPLY
# ===========================================================
elif page == "🎓 Course Supply":
    st.title("🎓 Course Supply - Analisis Coursera")
    st.caption("Ketersediaan course untuk menutup kesenjangan skill di pasar kerja IT")

    tab1, tab2, tab3 = st.tabs(["📊 Overview Course", "🔄 Supply vs Demand", "📋 Rekomendasi per Role"])

    with tab1:
        # KPI course
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Course", f"{len(df_c):,}")
        c2.metric("Role Tercakup", df_c['role_category'].nunique())
        c3.metric("Skill Unik (Course)", f"{len(course_skill_ctr):,}")
        c4.metric("Avg Skill/Course", f"{df_c['skill_count'].mean():.1f}")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribusi Course per Role")
            cc = df_c['role_category'].value_counts().head(15).reset_index()
            cc.columns = ['role', 'count']
            fig = px.bar(cc, x='count', y='role', orientation='h',
                         color='count', color_continuous_scale='Purples', text='count')
            fig.update_traces(textposition='outside')
            fig.update_layout(barmode='group', height=max(500, top_n_compare * 25),
                  title=f"Top {top_n_compare} Skill: Demand vs Supply", **PLOTLY_THEME)
            fig.update_yaxes(categoryorder='array', categoryarray=gap_df.sort_values('demand')['skill'].tolist())
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Distribusi Difficulty")
            diff = df_c['difficulty'].value_counts().reset_index()
            diff.columns = ['level', 'count']
            fig2 = px.pie(diff, values='count', names='level', hole=0.5,
                         color_discrete_sequence=['#16A34A', '#2563EB', '#D97706', '#DC2626'])
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            fig2.update_layout(height=220, **PLOTLY_THEME)
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Distribusi Durasi")
            dur = df_c['duration'].value_counts().reset_index()
            dur.columns = ['durasi', 'count']
            dur['durasi'] = dur['durasi'].str.replace('_', ' ').str.title()
            fig3 = px.bar(dur, x='durasi', y='count', color='count',
                          color_continuous_scale='Blues', text='count')
            fig3.update_traces(textposition='outside')
            fig3.update_layout(height=220, coloraxis_showscale=False, **PLOTLY_THEME)
            fig3.update_xaxes(tickangle=-25)
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.subheader("Skill Supply (Course) vs Demand (Loker)")
        st.info("Skill dengan **demand tinggi** tapi **supply rendah** adalah peluang terbesar untuk pembuatan course baru atau prioritas pembelajaran.")

        # Hitung overlap
        top_n_compare = st.slider("Jumlah skill teratas", 15, 50, 30)
        top_demand = {s: c for s, c in job_skill_ctr.most_common(top_n_compare)}
        top_supply = {s: c for s, c in course_skill_ctr.most_common(200)}

        gap_data = []
        for skill, demand_count in top_demand.items():
            supply_count = top_supply.get(skill, 0)
            gap_data.append({
                'skill': skill,
                'demand': demand_count,
                'supply': supply_count,
                'gap_ratio': demand_count / max(supply_count, 1),
            })
        gap_df = pd.DataFrame(gap_data).sort_values('gap_ratio', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            # Grouped bar: demand vs supply
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Demand (Loker)', y=gap_df['skill'], x=gap_df['demand'],
                                 orientation='h', marker_color='#2563EB'))
            fig.add_trace(go.Bar(name='Supply (Course)', y=gap_df['skill'], x=gap_df['supply'],
                                 orientation='h', marker_color='#7C3AED'))
            fig.update_layout(barmode='group', height=max(500, top_n_compare * 25),
                               yaxis={'categoryorder': 'array',
                                      'categoryarray': gap_df.sort_values('demand')['skill'].tolist()},
                               title=f"Top {top_n_compare} Skill: Demand vs Supply",
                               **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Skill Gap Terbesar")
            st.caption("Skill dengan rasio demand/supply tertinggi (banyak dicari, sedikit course)")
            gap_top = gap_df.head(15)
            fig2 = px.bar(gap_top, x='gap_ratio', y='skill', orientation='h',
                          color='gap_ratio', color_continuous_scale='Reds',
                          text=gap_top['gap_ratio'].round(1))
            fig2.update_traces(texttemplate='%{text:.1f}x', textposition='outside')
            fig2.update_layout(height=max(400, 15 * 30), coloraxis_showscale=False,
                               yaxis={'categoryorder': 'total ascending'},
                               xaxis_title='Demand/Supply Ratio',
                               title="Gap Ratio (Demand / Supply)", **PLOTLY_THEME)
            st.plotly_chart(fig2, use_container_width=True)

        # Skill kategorisasi
        st.divider()
        st.subheader("Kategorisasi Skill")
        col_m, col_g, col_s = st.columns(3)

        high_demand_skills = set(s for s, _ in job_skill_ctr.most_common(30))
        high_supply_skills = set(s for s, _ in course_skill_ctr.most_common(30))

        well_served = high_demand_skills & high_supply_skills
        underserved = high_demand_skills - high_supply_skills
        oversupplied = high_supply_skills - high_demand_skills

        with col_m:
            st.markdown("**Terlayani Baik** (demand & supply tinggi)")
            for s in sorted(well_served):
                st.markdown(f"<span class='skill-tag skill-tag-ok'>{s}</span>", unsafe_allow_html=True)

        with col_g:
            st.markdown("**Kurang Course** (demand tinggi, supply rendah)")
            for s in sorted(underserved):
                st.markdown(f"<span class='skill-tag skill-tag-gap'>{s}</span>", unsafe_allow_html=True)

        with col_s:
            st.markdown("**Oversupplied** (banyak course, demand lebih rendah)")
            for s in sorted(oversupplied):
                st.markdown(f"<span class='skill-tag'>{s}</span>", unsafe_allow_html=True)

    with tab3:
        st.subheader("Course Tersedia per Role IT")
        role_sel = st.selectbox("Pilih Role", sorted(df['role_label'].unique()), key='course_role')

        # Cari course yang match (pakai role_category)
        # Mapping role_label ke kemungkinan role_category di Coursera
        df_c_role = df_c[df_c['role_category'] == role_sel]

        if df_c_role.empty:
            # Coba partial match
            df_c_role = df_c[df_c['role_category'].str.contains(role_sel.split()[0], case=False, na=False)]

        col1, col2 = st.columns([2, 1])

        with col1:
            if not df_c_role.empty:
                st.success(f"Ditemukan **{len(df_c_role)} course** untuk role **{role_sel}**")
                for _, row in df_c_role.head(15).iterrows():
                    diff_color = {'BEGINNER': '#16A34A', 'INTERMEDIATE': '#D97706',
                                  'ADVANCED': '#DC2626', 'MIXED': '#2563EB'}.get(
                                      str(row.get('difficulty', '')), '#94A3B8')
                    dur_str = str(row.get('duration', '')).replace('_', ' ').title()
                    skills_str = ', '.join(row['skills_list'][:6])
                    if len(row['skills_list']) > 6:
                        skills_str += f" +{len(row['skills_list'])-6} lainnya"

                    st.markdown(f"""
                    <div class='insight-card'>
                        <div style='font-size:15px; font-weight:700; color:#F1F5F9;'>
                            <a href='{row["url"]}' target='_blank' style='color:#60A5FA; text-decoration:none;'>{row['name']}</a>
                        </div>
                        <div style='color:#94A3B8; font-size:12px; margin-top:4px;'>
                            {row['partner']} | <span style='color:{diff_color};'>{row.get('difficulty','')}</span> | {dur_str}
                        </div>
                        <div style='margin-top:8px; font-size:12px; color:#CBD5E1;'>
                            Skills: {skills_str}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"Belum ada course Coursera yang ter-mapping ke role '{role_sel}'.")

        with col2:
            # Perbandingan skill loker vs course untuk role ini
            job_role_skills = []
            df[df['role_label'] == role_sel]['skills_list'].apply(lambda l: job_role_skills.extend(l))
            job_role_ctr = Counter(job_role_skills)
            top_job = set(s for s, _ in job_role_ctr.most_common(10))

            course_role_skills = []
            df_c_role['skills_list'].apply(lambda l: course_role_skills.extend(l))
            course_role_ctr = Counter(course_role_skills)
            top_course = set(s for s, _ in course_role_ctr.most_common(10))

            covered = top_job & top_course
            uncovered = top_job - top_course

            st.markdown(f"**Top 10 Skill {role_sel}**")
            st.metric("Di-cover Course", f"{len(covered)}/{len(top_job)}")
            st.markdown("**Tercakup:**")
            for s in sorted(covered):
                st.markdown(f"<span class='skill-tag skill-tag-ok'>{s}</span>", unsafe_allow_html=True)
            if uncovered:
                st.markdown("**Belum tercakup:**")
                for s in sorted(uncovered):
                    st.markdown(f"<span class='skill-tag skill-tag-gap'>{s}</span>", unsafe_allow_html=True)

# ===========================================================
# PAGE 5: A/B TESTING
# ===========================================================
elif page == "🧪 A/B Testing":
    st.title("🧪 A/B Testing - Strategi Rekomendasi")
    st.caption("Eksperimen: Strategi A (Top-3 Heuristik) vs Strategi B (Cosine Similarity)")

    st.info("""
    **Konteks QLOP:** Modul AI Engine perlu memilih strategi terbaik untuk memetakan skill CV ke role.
    Eksperimen ini membandingkan dua pendekatan dan menentukan mana yang lebih akurat.

    - **H0:** Tidak ada perbedaan signifikan antara Strategi A dan B
    - **H1:** Strategi B (Cosine Similarity) lebih akurat
    - **alpha = 0.05** | Metrik: Precision@1 | N = test set 20%
    """)

    @st.cache_data(ttl=3600)
    def run_ab_test():
        from sklearn.model_selection import train_test_split
        from scipy import stats as scipy_stats

        df_ab = pd.read_csv("data/processed/MASTERED_DATA_FINAL_MODELING.csv")
        df_ab['skills_list'] = df_ab['hard_skills'].apply(
            lambda x: [s.strip() for s in str(x).split(',') if s.strip()] if pd.notna(x) else []
        )
        all_s = []
        df_ab['skills_list'].apply(lambda l: all_s.extend(l))
        freq = Counter(all_s)
        sel = sorted([s for s, c in freq.items() if c >= 20])

        mlb2 = MultiLabelBinarizer(classes=sel)
        X_all = mlb2.fit_transform(df_ab['skills_list'])

        df_train, df_test = train_test_split(
            df_ab, test_size=0.2, random_state=42, stratify=df_ab['role_label']
        )
        X_train = mlb2.transform(df_train['skills_list'])
        X_test  = mlb2.transform(df_test['skills_list'])
        y_test  = df_test['role_label'].values

        tr_mat = pd.DataFrame(X_train, columns=mlb2.classes_)
        tr_mat['role_label'] = df_train['role_label'].values
        role_prof = tr_mat.groupby('role_label').mean()
        role_order = role_prof.index.tolist()

        # Strategi A: Top-3 heuristik
        role_top3 = {r: set(role_prof.loc[r].nlargest(3).index) for r in role_order}
        def pred_a(skills):
            best, best_s = None, -1
            for r, t3 in role_top3.items():
                s = len(set(skills) & t3)
                if s > best_s:
                    best_s, best = s, r
            return best or 'Software Engineer'
        pred_A = [pred_a(df_test['skills_list'].iloc[i]) for i in range(len(df_test))]

        # Strategi B: Cosine Similarity
        rp_mat = role_prof.values
        pred_B = [role_order[np.argmax(cosine_similarity(X_test[i:i+1], rp_mat)[0])]
                  for i in range(len(X_test))]

        acc_A = np.mean(np.array(pred_A) == y_test)
        acc_B = np.mean(np.array(pred_B) == y_test)

        c_A = (np.array(pred_A) == y_test).astype(int)
        c_B = (np.array(pred_B) == y_test).astype(int)
        b = ((c_A == 1) & (c_B == 0)).sum()
        c = ((c_A == 0) & (c_B == 1)).sum()
        stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
        p_val = scipy_stats.chi2.sf(stat, df=1)

        np.random.seed(42)
        diffs = []
        for _ in range(3000):
            idx = np.random.choice(len(c_A), len(c_A), replace=True)
            diffs.append(c_B[idx].mean() - c_A[idx].mean())
        diffs = np.array(diffs)

        return acc_A, acc_B, p_val, stat, diffs, len(y_test)

    with st.spinner("Menjalankan A/B test..."):
        acc_A, acc_B, p_val, mcn_stat, boot_diffs, n_test = run_ab_test()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategi A", f"{acc_A*100:.2f}%", help="Top-3 Heuristik")
    c2.metric("Strategi B", f"{acc_B*100:.2f}%",
              delta=f"{(acc_B-acc_A)*100:+.2f}%", help="Cosine Similarity")
    c3.metric("McNemar p-value", f"{p_val:.4f}")
    c4.metric("Keputusan", "Tolak H0" if p_val < 0.05 else "Gagal Tolak H0",
              delta="Signifikan" if p_val < 0.05 else "Tidak Signifikan")

    st.divider()
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.subheader("Perbandingan Akurasi")
        fig = go.Figure(go.Bar(
            x=['Strategi A\n(Top-3 Heuristik)', 'Strategi B\n(Cosine Similarity)'],
            y=[acc_A * 100, acc_B * 100],
            text=[f"{acc_A*100:.2f}%", f"{acc_B*100:.2f}%"],
            textposition='outside',
            marker_color=['#64748B', '#2563EB'],
            marker_line_color='white', marker_line_width=1.5
        ))
        fig.update_layout(height=350, yaxis_range=[0, max(acc_A, acc_B) * 130],
                          yaxis_title='Precision@1 (%)', **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col_v2:
        st.subheader("Bootstrap Distribution (B - A)")
        ci_lo = np.percentile(boot_diffs, 2.5) * 100
        ci_hi = np.percentile(boot_diffs, 97.5) * 100
        mean_d = boot_diffs.mean() * 100

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=boot_diffs * 100, nbinsx=50,
                                    marker_color='#2563EB', opacity=0.7))
        fig2.add_vline(x=mean_d, line_dash='dash', line_color='red',
                       annotation_text=f"Mean: {mean_d:+.2f}%")
        fig2.add_vline(x=ci_lo, line_dash='dot', line_color='orange',
                       annotation_text=f"CI: [{ci_lo:+.1f}%")
        fig2.add_vline(x=ci_hi, line_dash='dot', line_color='orange',
                       annotation_text=f"{ci_hi:+.1f}%]")
        fig2.add_vline(x=0, line_color='white', opacity=0.3)
        fig2.update_layout(height=350, xaxis_title='Perbedaan Akurasi B-A (%)',
                           yaxis_title='Frekuensi', **PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"95% CI: [{ci_lo:+.2f}%, {ci_hi:+.2f}%]")

    st.divider()
    st.subheader("Kesimpulan")
    if p_val < 0.05:
        winner = "Strategi B (Cosine Similarity)" if acc_B > acc_A else "Strategi A (Top-3)"
        st.success(f"""
        **H0 ditolak** (p = {p_val:.4f} < 0.05)

        Terdapat perbedaan **signifikan** antara kedua strategi.
        **{winner}** terbukti lebih baik.

        Perbedaan: {(acc_B-acc_A)*100:+.2f}% (95% CI: [{ci_lo:+.2f}%, {ci_hi:+.2f}%])

        **Implikasi untuk QLOP:** AI Engine sebaiknya menggunakan {winner} untuk memetakan skill CV ke role.
        """)
    else:
        st.warning(f"""
        **Gagal tolak H0** (p = {p_val:.4f} >= 0.05)

        Tidak ada perbedaan signifikan. Meski Strategi B {'lebih' if acc_B > acc_A else 'kurang'} akurat
        ({(acc_B-acc_A)*100:+.2f}%), perbedaan bisa karena variasi sampling.

        **Implikasi untuk QLOP:** Gunakan Strategi B (Cosine Similarity) karena lebih scalable
        dan memanfaatkan seluruh dimensi skill, meskipun keunggulannya belum signifikan secara statistik.
        """)
