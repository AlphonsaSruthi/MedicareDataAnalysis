# ============================================================
# Medicare Analytics — SHAP Explainability App
# DSCI 5260 — Group 6
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Medicare SHAP | Group 6",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.header-bar {
    background: linear-gradient(135deg, #0a1628 0%, #112244 60%, #1a3a5c 100%);
    padding: 2rem 2.5rem 1.5rem; border-radius: 12px;
    margin-bottom: 1.5rem; border-left: 5px solid #2196F3;
}
.header-bar h1 { color: #ffffff; font-size: 1.9rem; font-weight: 700; margin: 0; }
.section-heading {
    font-size: 1.25rem; font-weight: 800; color: #1565C0;
    border-bottom: 2.5px solid #1565C0; padding-bottom: 0.45rem;
    margin: 1.6rem 0 1rem 0;
}
.key-insight-box {
    background: #e8f4fd; border-left: 4px solid #1565C0;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem;
    font-size: 0.95rem; color: #0d1f35; margin: 0.8rem 0; font-weight: 500;
}
.info-box {
    background: #fff8e1; border-left: 4px solid #f9a825;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem;
    font-size: 0.85rem; color: #4a3700; margin: 0.8rem 0;
}
section[data-testid="stSidebar"] { background: #0a1628 !important; min-width: 260px !important; }
section[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
section[data-testid="stSidebar"] hr { border-color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ── File IDs ────────────────────────────────────────────────
GDRIVE_IDS = {
    'rq2_shap_v': '1clJTycU_WGrb7xE1oL8w3L-0k5EOvI6U',
    'rq2_shap_b': '10kcgLxbpnDApVGT9lKOHN-c2fVEQNCGS',
    'rq2_shap_x': '1hqk74K_C_aep6hzq7oNGiJunveqIEhdD',
    'rq3_shap_v': '1WB77ZqytGwb_uqOYQ0PaZ_Na0uukm7R2',
    'rq3_shap_b': '1HBxE7feMGxB8WTi0CBEcBrmfzmEFrfCO',
    'rq3_shap_x': '1kfDDbB7hYvwBW9Ny3CgG4mkCF5sOLVJ6',
}
LOCAL_PATHS = {
    'rq2_shap_v': 'outputs/rq2_shap_values.npy',
    'rq2_shap_b': 'outputs/rq2_shap_base.npy',
    'rq2_shap_x': 'outputs/rq2_shap_X.parquet',
    'rq3_shap_v': 'outputs/rq3_shap_values.npy',
    'rq3_shap_b': 'outputs/rq3_shap_base.npy',
    'rq3_shap_x': 'outputs/rq3_shap_X.parquet',
}

def load_file(key, use_gdrive):
    if use_gdrive:
        try:
            import gdown
            ext   = LOCAL_PATHS[key].split('.')[-1]
            local = f"/tmp/{key}_cached.{ext}"
            if not os.path.exists(local):
                gdown.download(f"https://drive.google.com/uc?id={GDRIVE_IDS[key]}", local, quiet=True)
            return local
        except Exception as e:
            st.warning(f"Drive load failed for {key}: {e}")
    return LOCAL_PATHS[key]

@st.cache_data(show_spinner=False)
def load_shap_files(use_gdrive):
    rq2_sv   = np.load(load_file('rq2_shap_v', use_gdrive))
    rq2_base = np.load(load_file('rq2_shap_b', use_gdrive))[0]
    rq2_X    = pd.read_parquet(load_file('rq2_shap_x', use_gdrive))
    rq3_sv   = np.load(load_file('rq3_shap_v', use_gdrive))
    rq3_base = np.load(load_file('rq3_shap_b', use_gdrive))[0]
    rq3_X    = pd.read_parquet(load_file('rq3_shap_x', use_gdrive))
    return rq2_sv, rq2_base, rq2_X, rq3_sv, rq3_base, rq3_X

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Medicare SHAP")
    st.markdown("---")
    is_cloud = not os.path.exists('outputs/rq2_shap_values.npy')
    use_gdrive = st.toggle("☁️ Load from Google Drive", value=is_cloud)
    st.markdown("---")
    st.markdown("**What is SHAP?**")
    st.markdown("SHAP values show which features push each prediction higher or lower — and by how much.")
    st.markdown("---")
    st.caption("DSCI 5260 · Group 6")

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <h1>🧠 Model Explainability — SHAP Analysis</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
SHAP (SHapley Additive exPlanations) shows <strong>why</strong> the model makes each prediction —
which features push predictions up or down and by how much.
</div>
""", unsafe_allow_html=True)

# ── Load ────────────────────────────────────────────────────
with st.spinner("Loading SHAP files..."):
    try:
        rq2_sv, rq2_base, rq2_X, rq3_sv, rq3_base, rq3_X = load_shap_files(use_gdrive)
    except Exception as e:
        st.error(f"Could not load SHAP files: {e}")
        st.stop()

# ── Tabs ────────────────────────────────────────────────────
shap_tab1, shap_tab2 = st.tabs(["  RQ2 — Discharge Volume  ", "  RQ3 — Reimbursement  "])

# ── RQ2 SHAP ────────────────────────────────────────────────
with shap_tab1:
    st.markdown('<div class="section-heading">RQ2 — Feature Importance (Discharge Volume)</div>', unsafe_allow_html=True)

    mean_abs_shap = np.abs(rq2_sv).mean(axis=0)
    feat_imp = pd.DataFrame({
        'Feature': rq2_X.columns,
        'Mean |SHAP|': mean_abs_shap
    }).sort_values('Mean |SHAP|', ascending=False)
    feat_imp['Share %'] = (feat_imp['Mean |SHAP|'] / feat_imp['Mean |SHAP|'].sum() * 100).round(1)

    col_sh1, col_sh2 = st.columns([1.4, 1])
    with col_sh1:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        colors = ['#1565C0' if i < 3 else '#90caf9' for i in range(len(feat_imp))]
        bars = ax.barh(feat_imp['Feature'], feat_imp['Mean |SHAP|'], color=colors, edgecolor='white')
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('RQ2 — Feature Importance (XGBoost + SHAP)', fontsize=10, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        for bar, v in zip(bars, feat_imp['Mean |SHAP|']):
            ax.text(v + 0.001, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_sh2:
        st.markdown("**Feature Influence Share**")
        display_df = feat_imp[['Feature','Mean |SHAP|','Share %']].copy()
        display_df['Mean |SHAP|'] = display_df['Mean |SHAP|'].round(4)
        display_df['Share %']     = display_df['Share %'].round(1).astype(str) + '%'
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("#### SHAP Beeswarm Plot")
    shap.summary_plot(rq2_sv, rq2_X, show=False, plot_size=(10, 5))
    fig_bees = plt.gcf()
    fig_bees.suptitle('RQ2 — SHAP Beeswarm (each dot = one hospital-DRG pair)', fontsize=10, fontweight='bold', y=1.01)
    st.pyplot(fig_bees); plt.close()

    st.markdown("- **drg_te** accounts for ~45% of total model influence")
    st.markdown("- **hosp_te** accounts for ~23% | **DRG_Weight** is weak for volume prediction")
    st.markdown("- Geography and ownership have minimal predictive impact")

    st.markdown("#### Model Error by Subgroup (Test Set — 2023)")
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**By Geography**")
        st.dataframe(pd.DataFrame({
            'RUCA Group':      ['Metropolitan','Micropolitan','Small Town','Rural'],
            'MAE (discharges)':[12.01, 9.67, 8.53, 8.78],
            'MAPE (%)':        [31.4, 30.3, 31.3, 30.7],
            'Bias':            [5.29, 3.55, 2.76, 2.82]
        }), use_container_width=True, hide_index=True)
    with ec2:
        st.markdown("**By Ownership**")
        st.dataframe(pd.DataFrame({
            'Ownership':       ['For-Profit','Government','Non-Profit'],
            'MAE (discharges)':[11.98, 11.41, 11.05],
            'MAPE (%)':        [31.1, 32.0, 31.7],
            'Bias':            [5.34, 4.61, 4.46]
        }), use_container_width=True, hide_index=True)

# ── RQ3 SHAP ────────────────────────────────────────────────
with shap_tab2:
    st.markdown('<div class="section-heading">RQ3 — Feature Importance (Medicare Reimbursement)</div>', unsafe_allow_html=True)

    mean_abs_shap3 = np.abs(rq3_sv).mean(axis=0)
    feat_imp3 = pd.DataFrame({
        'Feature': rq3_X.columns,
        'Mean |SHAP|': mean_abs_shap3
    }).sort_values('Mean |SHAP|', ascending=False)
    feat_imp3['Share %'] = (feat_imp3['Mean |SHAP|'] / feat_imp3['Mean |SHAP|'].sum() * 100).round(1)

    col_sh3, col_sh4 = st.columns([1.4, 1])
    with col_sh3:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        colors3 = ['#2E7D32' if i == 0 else '#66bb6a' if i < 3 else '#a5d6a7' for i in range(len(feat_imp3))]
        bars3 = ax.barh(feat_imp3['Feature'], feat_imp3['Mean |SHAP|'], color=colors3, edgecolor='white')
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('RQ3 — Feature Importance (XGBoost + SHAP)', fontsize=10, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        for bar, v in zip(bars3, feat_imp3['Mean |SHAP|']):
            ax.text(v + 0.001, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_sh4:
        st.markdown("**Feature Influence Share**")
        display_df3 = feat_imp3[['Feature','Mean |SHAP|','Share %']].copy()
        display_df3['Mean |SHAP|'] = display_df3['Mean |SHAP|'].round(4)
        display_df3['Share %']     = display_df3['Share %'].round(1).astype(str) + '%'
        st.dataframe(display_df3, use_container_width=True, hide_index=True)

    st.markdown("#### SHAP Beeswarm Plot")
    shap.summary_plot(rq3_sv, rq3_X, show=False, plot_size=(10, 5))
    fig_bees3 = plt.gcf()
    fig_bees3.suptitle('RQ3 — SHAP Beeswarm (each dot = one hospital-DRG pair)', fontsize=10, fontweight='bold', y=1.01)
    st.pyplot(fig_bees3); plt.close()

    st.markdown("- **DRG_Weight** accounts for **71.2%** of total model influence")
    st.markdown("- **BED_CNT** = 12.7% | DRG_Weight + BED_CNT together explain **84%**")
    st.markdown("- All RUCA geography groups combined = only **~1.6%**")

    st.markdown("#### Model Error by Subgroup (Test Set)")
    ec3, ec4 = st.columns(2)
    with ec3:
        st.markdown("**By Geography**")
        st.dataframe(pd.DataFrame({
            'RUCA Group': ['Metropolitan','Micropolitan','Small Town','Rural'],
            'MAE ($)':    [2315, 1163, 1177, 870],
            'MAPE (%)':   [14.6, 11.8, 11.5, 9.6],
            'Bias ($)':   [375, 139, 131, 129]
        }), use_container_width=True, hide_index=True)
    with ec4:
        st.markdown("**By Ownership**")
        st.dataframe(pd.DataFrame({
            'Ownership':  ['For-Profit','Government','Non-Profit'],
            'MAE ($)':    [2340, 2102, 1609],
            'MAPE (%)':   [15.2, 12.6, 11.7],
            'Bias ($)':   [370, 412, 229]
        }), use_container_width=True, hide_index=True)
