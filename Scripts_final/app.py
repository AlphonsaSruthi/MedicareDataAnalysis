# ============================================================
# Medicare Inpatient Analytics Dashboard
# DSCI 5260 — Group 6 | Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import shap
import warnings
import os
warnings.filterwarnings('ignore')

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Medicare Analytics | Group 6",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme / CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark navy header bar */
.header-bar {
    background: linear-gradient(135deg, #0a1628 0%, #112244 60%, #1a3a5c 100%);
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 5px solid #2196F3;
}
.header-bar h1 {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.header-bar p {
    color: #90b4d4;
    font-size: 0.9rem;
    margin: 0;
    font-family: 'IBM Plex Mono', monospace;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.metric-card {
    background: #f0f4f9;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    flex: 1;
    min-width: 140px;
    border-top: 3px solid #2196F3;
}
.metric-card.green  { border-top-color: #2E7D32; }
.metric-card.orange { border-top-color: #EF6C00; }
.metric-card.red    { border-top-color: #C62828; }
.metric-card.teal   { border-top-color: #00838F; }

.metric-card .label {
    font-size: 0.95rem;
    color: #607080;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #0d1f35;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-card .sub {
    font-size: 0.85rem;
    color: #607080;
    margin-top: 0.25rem;
}

/* Section headings */
.section-heading {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0d1f35;
    border-bottom: 2px solid #2196F3;
    padding-bottom: 0.4rem;
    margin: 1.4rem 0 0.9rem 0;
}

/* Prediction result box */
.pred-box {
    background: linear-gradient(135deg, #e8f4fd, #f0f8ff);
    border: 1.5px solid #90caf9;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin: 0.8rem 0;
}
.pred-box .pred-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #1565C0;
    font-weight: 600;
}
.pred-box .pred-value {
    font-size: 2.4rem;
    font-weight: 700;
    color: #0d47a1;
    font-family: 'IBM Plex Mono', monospace;
}
.pred-box .pred-sub {
    font-size: 0.8rem;
    color: #455a64;
    margin-top: 0.3rem;
}

/* Insight pill */
.insight-pill {
    display: inline-block;
    background: #e3f2fd;
    color: #1565C0;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.2rem 0.2rem 0.2rem 0;
}
.insight-pill.green  { background: #e8f5e9; color: #2E7D32; }
.insight-pill.orange { background: #fff3e0; color: #E65100; }
.insight-pill.red    { background: #fce4ec; color: #b71c1c; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a1628 !important;
}
section[data-testid="stSidebar"] * {
    color: #c8d8e8 !important;
}
section[data-testid="stSidebar"] .sidebar-title {
    color: #ffffff !important;
    font-size: 1.1rem;
    font-weight: 700;
}
section[data-testid="stSidebar"] hr {
    border-color: #1e3a5f !important;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.3px !important;
}

/* Info box */
.info-box {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    font-size: 0.85rem;
    color: #4a3700;
    margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING — Google Drive OR Local
# ============================================================

# ── File ID mapping — replace with your actual Google Drive file IDs ──
GDRIVE_IDS = {
    'data':         'YOUR_DATA_FILE_ID',        # df_medidata_clean.parquet
    'rq2_model':    'YOUR_RQ2_MODEL_FILE_ID',   # rq2_xgb_model.pkl
    'rq3_model':    'YOUR_RQ3_MODEL_FILE_ID',   # rq3_xgb_model.pkl
    'hosp_te':      'YOUR_HOSP_TE_FILE_ID',     # hosp_te_lookup.pkl
    'drg_te':       'YOUR_DRG_TE_FILE_ID',      # drg_te_lookup.pkl
    'rq2_shap_v':   'YOUR_RQ2_SHAP_V_FILE_ID',  # rq2_shap_values.npy
    'rq2_shap_b':   'YOUR_RQ2_SHAP_B_FILE_ID',  # rq2_shap_base.npy
    'rq2_shap_x':   'YOUR_RQ2_SHAP_X_FILE_ID',  # rq2_shap_X.parquet
    'rq3_shap_v':   'YOUR_RQ3_SHAP_V_FILE_ID',  # rq3_shap_values.npy
    'rq3_shap_b':   'YOUR_RQ3_SHAP_B_FILE_ID',  # rq3_shap_base.npy
    'rq3_shap_x':   'YOUR_RQ3_SHAP_X_FILE_ID',  # rq3_shap_X.parquet
    'rq2_forecast': 'YOUR_RQ2_FORECAST_FILE_ID', # RQ2_Predictions_2024_WithCI.csv
    'rq3_forecast': 'YOUR_RQ3_FORECAST_FILE_ID', # RQ3_Predictions_2024.csv
}

# ── Local paths — used when running on your own machine ──
LOCAL_PATHS = {
    'data':         '../Data/Processed_Data/df_medidata_clean.parquet',
    'rq2_model':    '../Data/Processed_Data/rq2_xgb_model.pkl',
    'rq3_model':    '../Data/Processed_Data/rq3_xgb_model.pkl',
    'hosp_te':      '../Data/Processed_Data/hosp_te_lookup.pkl',
    'drg_te':       '../Data/Processed_Data/drg_te_lookup.pkl',
    'rq2_shap_v':   'outputs/rq2_shap_values.npy',
    'rq2_shap_b':   'outputs/rq2_shap_base.npy',
    'rq2_shap_x':   'outputs/rq2_shap_X.parquet',
    'rq3_shap_v':   'outputs/rq3_shap_values.npy',
    'rq3_shap_b':   'outputs/rq3_shap_base.npy',
    'rq3_shap_x':   'outputs/rq3_shap_X.parquet',
    'rq2_forecast': '../Data/Processed_Data/RQ2_Predictions_2024_WithCI.csv',
    'rq3_forecast': '../Data/Processed_Data/RQ3_Predictions_2024.csv',
}

def gdrive_url(file_id):
    return f"https://drive.google.com/uc?id={file_id}"

def load_file(key, use_gdrive=False):
    """Load a file from Google Drive or local path."""
    if use_gdrive:
        try:
            import gdown
            fid  = GDRIVE_IDS[key]
            fname = key.replace('/', '_') + '_cached'
            ext   = LOCAL_PATHS[key].split('.')[-1]
            local = f"/tmp/{fname}.{ext}"
            if not os.path.exists(local):
                gdown.download(gdrive_url(fid), local, quiet=True)
            return local
        except Exception as e:
            st.warning(f"Google Drive load failed for {key}: {e}. Falling back to local.")
    return LOCAL_PATHS[key]

@st.cache_data(show_spinner=False)
def load_data(use_gdrive):
    path = load_file('data', use_gdrive)
    df = pd.read_parquet(path)
    # Ensure derived columns exist
    if 'Payment_Gap' not in df.columns:
        df['Payment_Gap'] = df['Avg_Submtd_Cvrd_Chrg'] - df['Avg_Mdcr_Pymt_Amt']
    if 'Payment_Ratio' not in df.columns:
        df['Payment_Ratio'] = df['Avg_Mdcr_Pymt_Amt'] / df['Avg_Submtd_Cvrd_Chrg']
    if 'payment_gap_per_discharge' not in df.columns:
        df['payment_gap_per_discharge'] = df['Payment_Gap'] / df['Tot_Dschrgs']
    if 'RUCA_Group' not in df.columns:
        def ruca_group(v):
            if pd.isna(v) or v == 99: return 'Unknown'
            if v <= 3:  return 'Metropolitan'
            if v <= 6:  return 'Micropolitan'
            if v <= 9:  return 'Small Town'
            return 'Rural'
        df['RUCA_Group'] = df['Rndrng_Prvdr_RUCA'].apply(ruca_group)
    return df

@st.cache_resource(show_spinner=False)
def load_models(use_gdrive):
    rq2_model    = joblib.load(load_file('rq2_model',  use_gdrive))
    rq3_model    = joblib.load(load_file('rq3_model',  use_gdrive))
    hosp_te      = joblib.load(load_file('hosp_te',    use_gdrive))
    drg_te       = joblib.load(load_file('drg_te',     use_gdrive))
    return rq2_model, rq3_model, hosp_te, drg_te

@st.cache_data(show_spinner=False)
def load_shap(use_gdrive):
    rq2_sv  = np.load(load_file('rq2_shap_v', use_gdrive))
    rq2_base= np.load(load_file('rq2_shap_b', use_gdrive))[0]
    rq2_X   = pd.read_parquet(load_file('rq2_shap_x', use_gdrive))
    rq3_sv  = np.load(load_file('rq3_shap_v', use_gdrive))
    rq3_base= np.load(load_file('rq3_shap_b', use_gdrive))[0]
    rq3_X   = pd.read_parquet(load_file('rq3_shap_x', use_gdrive))
    return rq2_sv, rq2_base, rq2_X, rq3_sv, rq3_base, rq3_X

@st.cache_data(show_spinner=False)
def load_forecasts(use_gdrive):
    rq2_fc = pd.read_csv(load_file('rq2_forecast', use_gdrive))
    rq3_fc = pd.read_csv(load_file('rq3_forecast', use_gdrive))
    return rq2_fc, rq3_fc


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Medicare Analytics</div>', unsafe_allow_html=True)
    st.markdown("---")

    use_gdrive = st.toggle("☁️ Load from Google Drive", value=False,
                           help="Toggle ON to load files from Google Drive. OFF = local files.")

    st.markdown("---")
    st.markdown("**📦 Models**")
    st.markdown("""
- RQ2: XGBoost (Temporal Split)
- RQ3: XGBoost (Random Split)
    """)
    st.markdown("**📊 Performance**")
    st.markdown("""
| | R² | MAE |
|--|--|--|
| RQ2 | 0.682 | 12 discharges |
| RQ3 | 0.922 | $2,190 |
    """)
    st.markdown("---")
    st.markdown("**🗓️ Data Source**")
    st.markdown("CMS Medicare Inpatient  \n2017–2023 | ~1.18M records")
    st.markdown("---")
    st.caption("Built with Streamlit · Group 6")


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-bar">
    <h1>Medicare Inpatient Analytics Dashboard</h1>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ALL DATA
# ============================================================
with st.spinner("Loading data and models..."):
    try:
        df = load_data(use_gdrive)
        rq2_model, rq3_model, hosp_te_lookup, drg_te_lookup = load_models(use_gdrive)
        rq2_sv, rq2_base, rq2_X, rq3_sv, rq3_base, rq3_X   = load_shap(use_gdrive)
        rq2_fc, rq3_fc = load_forecasts(use_gdrive)
        data_ok = True
    except Exception as e:
        st.error(f"❌ Could not load files: {e}")
        st.info("Make sure all files exist at the paths defined in LOCAL_PATHS, or enable Google Drive and set correct file IDs.")
        data_ok = False
        st.stop()


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "  EDA — Data Exploration  ",
    "  Predict  ",
    "  SHAP — Explainability  ",
    "  2024 Forecast  "
])


# ============================================================
# TAB 1 — EDA
# ============================================================
with tab1:

    # ── Top KPI cards ──────────────────────────────────────
    st.markdown('<div class="section-heading">Dataset Overview</div>', unsafe_allow_html=True)

    total_records  = len(df)
    total_hospitals= df['Rndrng_Prvdr_CCN'].nunique()
    total_drgs     = df['DRG_Cd'].nunique()
    median_gap     = df['Payment_Gap'].median()
    median_ratio   = df['Payment_Ratio'].median()
    years_covered  = f"{int(df['Data_Year'].min())}–{int(df['Data_Year'].max())}"

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Total Records</div>
            <div class="value">{total_records:,.0f}</div>
            <div class="sub">Hospital-DRG-Year rows</div>
        </div>
        <div class="metric-card green">
            <div class="label">Unique Hospitals</div>
            <div class="value">{total_hospitals:,}</div>
            <div class="sub">Provider CCNs</div>
        </div>
        <div class="metric-card orange">
            <div class="label">Unique DRGs</div>
            <div class="value">{total_drgs:,}</div>
            <div class="sub">Diagnosis groups</div>
        </div>
        <div class="metric-card red">
            <div class="label">Median Payment Gap</div>
            <div class="value">${median_gap:,.0f}</div>
            <div class="sub">Billed minus Medicare paid</div>
        </div>
        <div class="metric-card teal">
            <div class="label">Median Payment Ratio</div>
            <div class="value">{median_ratio:.2f}</div>
            <div class="sub">Medicare pays {median_ratio*100:.0f}¢ per $1 billed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1: Discharge trend + Payment gap trend ─────────
    st.markdown('<div class="section-heading">National Trends (2017–2023)</div>', unsafe_allow_html=True)

    yearly = (df.groupby('Data_Year')
                .agg(Mean_Dschrgs  = ('Tot_Dschrgs', 'mean'),
                     Total_Dschrgs = ('Tot_Dschrgs', 'sum'),
                     Median_Gap    = ('Payment_Gap', 'median'),
                     Median_Ratio  = ('Payment_Ratio', 'median'))
                .reset_index())

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(yearly['Data_Year'], yearly['Mean_Dschrgs'],
                marker='o', color='#1565C0', linewidth=2.2, markersize=6)
        ax.axvspan(2019.5, 2021.5, alpha=0.10, color='#C62828', label='COVID window')
        ax.set_title('Mean Discharges per Hospital-DRG Pair', fontsize=11, fontweight='bold')
        ax.set_xlabel('Year'); ax.set_ylabel('Mean Discharges')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("- COVID dip clearly visible in 2020–2021")
        st.markdown("- Gradual national decline in discharge volume post-2021")

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(yearly['Data_Year'], yearly['Median_Gap'],
                marker='s', color='#C62828', linewidth=2.2, markersize=6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
        ax.set_title('Median Payment Gap Over Time', fontsize=11, fontweight='bold')
        ax.set_xlabel('Year'); ax.set_ylabel('Median Gap ($)')
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("- Payment gap widening over time")
        st.markdown("- Billed charges rise faster than Medicare payments")

    # ── Row 2: Ownership + Geography ──────────────────────
    st.markdown('<div class="section-heading">Payment Gap by Hospital Type</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    OWN_PAL  = {'For-Profit': '#1565C0', 'Non-Profit': '#EF6C00', 'Government': '#2E7D32'}
    RUCA_PAL = {'Metropolitan': '#1565C0', 'Micropolitan': '#EF6C00', 'Small Town': '#F9A825', 'Rural': '#2E7D32'}
    RUCA_ORD = ['Metropolitan', 'Micropolitan', 'Small Town', 'Rural']

    with col3:
        own_df = df[df['Ownership_Type'].isin(['For-Profit','Non-Profit','Government'])]
        metrics = own_df.groupby('Ownership_Type').agg(
            Payment_Ratio            = ('Payment_Ratio', 'median'),
            payment_gap_per_discharge= ('payment_gap_per_discharge', 'median'),
            discharges_per_bed       = ('discharges_per_bed', 'median') if 'discharges_per_bed' in df.columns else ('Tot_Dschrgs','median')
        ).reindex(['For-Profit','Non-Profit','Government'])

        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
        for ax, col_name, title, fmt in zip(
            axes,
            ['Payment_Ratio', 'payment_gap_per_discharge'],
            ['Payment Ratio\n(Medicare / Billed)', 'Gap per Discharge ($)'],
            ['{:.2f}', '${:,.0f}']
        ):
            vals  = metrics[col_name]
            bars  = ax.bar(vals.index, vals.values,
                           color=[OWN_PAL[o] for o in vals.index], width=0.5, edgecolor='white')
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.tick_params(axis='x', rotation=15, labelsize=8)
            ax.spines[['top','right']].set_visible(False)
            for bar, v in zip(bars, vals.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                        fmt.format(v), ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        fig.suptitle('Ownership Type Comparison', fontsize=10, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("- Non-Profit carries the largest gap per discharge: **$2,480**")
        st.markdown("- Non-Profit recovers only **13¢ per $1 billed**")

    with col4:
        geo_df = df[df['RUCA_Group'].isin(RUCA_ORD)]
        geo_metrics = geo_df.groupby('RUCA_Group').agg(
            Payment_Ratio            = ('Payment_Ratio', 'median'),
            payment_gap_per_discharge= ('payment_gap_per_discharge', 'median')
        ).reindex(RUCA_ORD)

        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
        for ax, col_name, title, fmt in zip(
            axes,
            ['Payment_Ratio', 'payment_gap_per_discharge'],
            ['Payment Ratio\n(Medicare / Billed)', 'Gap per Discharge ($)'],
            ['{:.2f}', '${:,.0f}']
        ):
            vals = geo_metrics[col_name]
            bars = ax.bar(vals.index, vals.values,
                          color=[RUCA_PAL[g] for g in vals.index], width=0.5, edgecolor='white')
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.tick_params(axis='x', rotation=20, labelsize=7.5)
            ax.spines[['top','right']].set_visible(False)
            for bar, v in zip(bars, vals.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                        fmt.format(v), ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        fig.suptitle('Geography (RUCA) Comparison', fontsize=10, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("- Rural hospitals recover **36¢ per $1 billed** vs Metropolitan **19¢**")
        st.markdown("- Metropolitan carries the **largest payment gap** per discharge")

    # ── Row 3: Discharge distribution + Payment ratio dist ─
    st.markdown('<div class="section-heading">Distribution Analysis</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)

    with col5:
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
        axes[0].hist(df['Tot_Dschrgs'].clip(upper=500), bins=50,
                     color='#1565C0', alpha=0.8, edgecolor='white')
        axes[0].set_title('Discharge Volume (Raw)', fontsize=9, fontweight='bold')
        axes[0].set_xlabel('Discharges'); axes[0].set_ylabel('Count')

        axes[1].hist(np.log1p(df['Tot_Dschrgs']), bins=50,
                     color='#EF6C00', alpha=0.8, edgecolor='white')
        axes[1].set_title('Discharge Volume (Log)', fontsize=9, fontweight='bold')
        axes[1].set_xlabel('log1p(Discharges)'); axes[1].set_ylabel('Count')
        fig.suptitle('RQ2 Target — Before & After Log Transform', fontsize=9, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col6:
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
        axes[0].hist(df['Avg_Mdcr_Pymt_Amt'].clip(upper=100000), bins=50,
                     color='#2E7D32', alpha=0.8, edgecolor='white')
        axes[0].set_title('Medicare Payment (Raw)', fontsize=9, fontweight='bold')
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
        axes[0].set_ylabel('Count')

        axes[1].hist(np.log1p(df['Avg_Mdcr_Pymt_Amt']), bins=50,
                     color='#00838F', alpha=0.8, edgecolor='white')
        axes[1].set_title('Medicare Payment (Log)', fontsize=9, fontweight='bold')
        axes[1].set_xlabel('log1p($)'); axes[1].set_ylabel('Count')
        fig.suptitle('RQ3 Target — Before & After Log Transform', fontsize=9, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Row 4: Top DRGs by payment gap ────────────────────
    st.markdown('<div class="section-heading">Top 15 DRGs by Median Payment Gap</div>', unsafe_allow_html=True)

    drg_gap = (df.groupby(['DRG_Cd','DRG_Desc'])
                 .agg(median_gap   = ('Payment_Gap', 'median'),
                      total_dschrg = ('Tot_Dschrgs', 'sum'))
                 .reset_index()
                 .sort_values('median_gap', ascending=False)
                 .head(15))
    drg_gap['Label'] = drg_gap['DRG_Cd'].astype(str) + ': ' + drg_gap['DRG_Desc'].str[:45]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    bars = ax.barh(drg_gap['Label'], drg_gap['median_gap'],
                   color='#1565C0', alpha=0.85)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
    ax.set_xlabel('Median Payment Gap')
    ax.set_title('Top 15 DRGs with Largest Medicare Payment Gap', fontsize=11, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x', alpha=0.25)
    for bar, v in zip(bars, drg_gap['median_gap']):
        ax.text(v + 500, bar.get_y()+bar.get_height()/2,
                f'${v/1000:.0f}K', va='center', fontsize=8)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Summary stats table ────────────────────────────────
    with st.expander("📋 Summary Statistics Table"):
        key_cols = ['Tot_Dschrgs','Avg_Submtd_Cvrd_Chrg','Avg_Mdcr_Pymt_Amt','Payment_Gap','Payment_Ratio']
        summary = df[key_cols].describe().round(2)
        summary = summary.rename(index={'50%':'median'})
        st.dataframe(summary, use_container_width=True)


# ============================================================
# TAB 2 — PREDICT
# ============================================================
with tab2:

    st.markdown('<div class="section-heading">Interactive Prediction — RQ2 & RQ3</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Enter hospital details below. The model will predict <strong>discharge volume (RQ2)</strong>
    and <strong>Medicare reimbursement (RQ3)</strong> using your trained XGBoost models.
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ─────────────────────────────────────────
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**🏥 Hospital Details**")
        hospital_ccn = st.number_input("Hospital CCN", value=370781, step=1,
                                        help="Provider CCN number from CMS data")
        drg_code     = st.number_input("DRG Code", value=470, step=1,
                                        help="MS-DRG procedure code (e.g. 470 = Knee Replacement)")
        bed_count    = st.slider("Number of Beds", min_value=10, max_value=3000,
                                  value=250, step=10)
        year         = st.selectbox("Prediction Year", [2024, 2023, 2022], index=0)

    with col_b:
        st.markdown("**📍 Hospital Classification**")
        ownership = st.radio("Ownership Type",
                              ['Non-Profit', 'For-Profit', 'Government'],
                              horizontal=True)
        location  = st.radio("Geographic Classification (RUCA)",
                              ['Metropolitan', 'Micropolitan', 'Small Town', 'Rural'],
                              horizontal=True)

        st.markdown("**📊 Auto-computed Inputs**")
        avg_hosp_te = hosp_te_lookup.mean()
        avg_drg_te  = drg_te_lookup.mean()
        avg_drg_wt  = df.groupby('DRG_Cd')['DRG_Weight'].mean().mean()

        hosp_te_val = hosp_te_lookup.get(hospital_ccn, avg_hosp_te)
        drg_te_val  = drg_te_lookup.get(drg_code, avg_drg_te)
        drg_wt_val  = df[df['DRG_Cd'] == drg_code]['DRG_Weight'].mean() if drg_code in df['DRG_Cd'].values else avg_drg_wt

        hosp_known = hospital_ccn in hosp_te_lookup.index
        drg_known  = drg_code in drg_te_lookup.index

        st.markdown(f"""
        | Signal | Value | Source |
        |---|---|---|
        | Hospital volume score (`hosp_te`) | `{hosp_te_val:.1f}` | {'✅ Known hospital' if hosp_known else '⚠️ New hospital — using average'} |
        | DRG demand score (`drg_te`) | `{drg_te_val:.1f}` | {'✅ Known DRG' if drg_known else '⚠️ New DRG — using average'} |
        | DRG complexity weight | `{drg_wt_val:.3f}` | CMS assigned weight |
        """)

    # ── Predict button ──────────────────────────────────────
    st.markdown("---")
    predict_btn = st.button("🔮 Run Prediction", type="primary", use_container_width=False)

    if predict_btn:
        # ── Build feature row ──
        rq2_row = pd.DataFrame([{
            'DRG_Weight'        : drg_wt_val,
            'BED_CNT'           : bed_count,
            'hosp_te'           : hosp_te_val,
            'drg_te'            : drg_te_val,
            'own_For-Profit'    : 1 if ownership == 'For-Profit'   else 0,
            'own_Non-Profit'    : 1 if ownership == 'Non-Profit'   else 0,
            'ruca_Metropolitan' : 1 if location  == 'Metropolitan' else 0,
            'ruca_Micropolitan' : 1 if location  == 'Micropolitan' else 0,
            'ruca_Small Town'   : 1 if location  == 'Small Town'   else 0,
            'Data_Year'         : year,
        }])

        tot_dschrg_hist = df[(df['Rndrng_Prvdr_CCN'] == hospital_ccn) &
                              (df['DRG_Cd'] == drg_code)]['Tot_Dschrgs'].mean()
        if np.isnan(tot_dschrg_hist):
            tot_dschrg_hist = df['Tot_Dschrgs'].mean()

        rq3_row = pd.DataFrame([{
            'DRG_Weight'          : drg_wt_val,
            'BED_CNT'             : bed_count,
            'Log_Tot_Dschrgs'     : np.log1p(tot_dschrg_hist),
            'own_For-Profit'      : 1 if ownership == 'For-Profit'   else 0,
            'own_Non-Profit'      : 1 if ownership == 'Non-Profit'   else 0,
            'ruca_Metropolitan'   : 1 if location  == 'Metropolitan' else 0,
            'ruca_Micropolitan'   : 1 if location  == 'Micropolitan' else 0,
            'ruca_Small Town'     : 1 if location  == 'Small Town'   else 0,
            'Data_Year'           : year,
            'outlier_payment_flag': 0,
        }])

        RQ2_FEATURES = ['DRG_Weight','BED_CNT','hosp_te','drg_te',
                         'own_For-Profit','own_Non-Profit',
                         'ruca_Metropolitan','ruca_Micropolitan','ruca_Small Town','Data_Year']
        RQ3_FEATURES = ['DRG_Weight','BED_CNT','Log_Tot_Dschrgs',
                         'own_For-Profit','own_Non-Profit',
                         'ruca_Metropolitan','ruca_Micropolitan','ruca_Small Town',
                         'Data_Year','outlier_payment_flag']

        pred_dschrg = int(np.expm1(rq2_model.predict(rq2_row[RQ2_FEATURES])[0]))
        pred_usd    = float(np.expm1(rq3_model.predict(rq3_row[RQ3_FEATURES])[0]))

        # ── Results ──
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">🏥 RQ2 — Predicted Discharge Volume</div>
                <div class="pred-value">{pred_dschrg:,}</div>
                <div class="pred-sub">discharges for Hospital {hospital_ccn} · DRG {drg_code} · {year}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Model: XGBoost Temporal Split | Test R² = 0.682 | MAE = 12 discharges")

        with res_col2:
            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">💰 RQ3 — Predicted Medicare Reimbursement</div>
                <div class="pred-value">${pred_usd:,.2f}</div>
                <div class="pred-sub">avg Medicare payment per discharge · {ownership} · {location}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Model: XGBoost Random Split | Test R² = 0.922 | MAE = $2,190")

        # ── Input summary ──
        with st.expander("📋 Full Input Summary"):
            st.dataframe(pd.DataFrame({
                'Input': ['Hospital CCN', 'DRG Code', 'Beds', 'Year',
                          'Ownership', 'Location', 'hosp_te', 'drg_te', 'DRG Weight'],
                'Value': [hospital_ccn, drg_code, bed_count, year,
                          ownership, location,
                          f'{hosp_te_val:.1f} ({"known" if hosp_known else "avg"})',
                          f'{drg_te_val:.1f} ({"known" if drg_known else "avg"})',
                          f'{drg_wt_val:.3f}']
            }), use_container_width=True, hide_index=True)


# ============================================================
# TAB 3 — SHAP
# ============================================================
with tab3:

    st.markdown('<div class="section-heading">Model Explainability — SHAP Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    SHAP (SHapley Additive exPlanations) shows <strong>why</strong> the model makes each prediction —
    which features push it up or down and by how much.
    </div>
    """, unsafe_allow_html=True)

    shap_tab1, shap_tab2 = st.tabs(["📊 RQ2 — Discharge Volume", "💰 RQ3 — Reimbursement"])

    # ── RQ2 SHAP ───────────────────────────────────────────
    with shap_tab1:

        # Feature importance bar chart
        st.markdown("#### Feature Importance (Mean |SHAP|)")
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
            bars = ax.barh(feat_imp['Feature'], feat_imp['Mean |SHAP|'],
                           color=colors, edgecolor='white')
            ax.invert_yaxis()
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title('RQ2 — Feature Importance (XGBoost + SHAP)', fontsize=10, fontweight='bold')
            ax.spines[['top','right']].set_visible(False)
            for bar, v in zip(bars, feat_imp['Mean |SHAP|']):
                ax.text(v + 0.001, bar.get_y()+bar.get_height()/2,
                        f'{v:.3f}', va='center', fontsize=8)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        with col_sh2:
            st.markdown("**Feature Influence Share**")
            display_df = feat_imp[['Feature','Mean |SHAP|','Share %']].copy()
            display_df['Mean |SHAP|'] = display_df['Mean |SHAP|'].round(4)
            display_df['Share %']     = display_df['Share %'].round(1).astype(str) + '%'
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Beeswarm plot
        st.markdown("#### SHAP Beeswarm Plot — How Feature Values Drive Predictions")
        shap.summary_plot(rq2_sv, rq2_X, show=False, plot_size=(10, 5))
        fig_bees = plt.gcf()
        fig_bees.suptitle('RQ2 — SHAP Beeswarm (each dot = one hospital-DRG pair)',
                           fontsize=10, fontweight='bold', y=1.01)
        st.pyplot(fig_bees); plt.close()

        st.markdown("- **drg_te** accounts for ~45% of total model influence")
        st.markdown("- **hosp_te** accounts for ~23% | **DRG_Weight** is weak for volume prediction")
        st.markdown("- Geography and ownership have minimal predictive impact")

        # Subgroup error table
        st.markdown("#### Model Error by Subgroup (Test Set — 2023)")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("**By Geography**")
            geo_err = pd.DataFrame({
                'RUCA Group':     ['Metropolitan','Micropolitan','Small Town','Rural'],
                'MAE (discharges)': [12.01, 9.67, 8.53, 8.78],
                'MAPE (%)':       [31.4, 30.3, 31.3, 30.7],
                'Bias (discharges)':[5.29, 3.55, 2.76, 2.82]
            })
            st.dataframe(geo_err, use_container_width=True, hide_index=True)
        with ec2:
            st.markdown("**By Ownership**")
            own_err = pd.DataFrame({
                'Ownership':   ['For-Profit','Government','Non-Profit'],
                'MAE (discharges)': [11.98, 11.41, 11.05],
                'MAPE (%)':   [31.1, 32.0, 31.7],
                'Bias':       [5.34, 4.61, 4.46]
            })
            st.dataframe(own_err, use_container_width=True, hide_index=True)

    # ── RQ3 SHAP ───────────────────────────────────────────
    with shap_tab2:

        st.markdown("#### Feature Importance (Mean |SHAP|)")
        mean_abs_shap3 = np.abs(rq3_sv).mean(axis=0)
        feat_imp3 = pd.DataFrame({
            'Feature': rq3_X.columns,
            'Mean |SHAP|': mean_abs_shap3
        }).sort_values('Mean |SHAP|', ascending=False)
        feat_imp3['Share %'] = (feat_imp3['Mean |SHAP|'] / feat_imp3['Mean |SHAP|'].sum() * 100).round(1)

        col_sh3, col_sh4 = st.columns([1.4, 1])

        with col_sh3:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            colors3 = ['#2E7D32' if i == 0 else '#66bb6a' if i < 3 else '#a5d6a7'
                       for i in range(len(feat_imp3))]
            bars3 = ax.barh(feat_imp3['Feature'], feat_imp3['Mean |SHAP|'],
                            color=colors3, edgecolor='white')
            ax.invert_yaxis()
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title('RQ3 — Feature Importance (XGBoost + SHAP)', fontsize=10, fontweight='bold')
            ax.spines[['top','right']].set_visible(False)
            for bar, v in zip(bars3, feat_imp3['Mean |SHAP|']):
                ax.text(v + 0.001, bar.get_y()+bar.get_height()/2,
                        f'{v:.3f}', va='center', fontsize=8)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        with col_sh4:
            st.markdown("**Feature Influence Share**")
            display_df3 = feat_imp3[['Feature','Mean |SHAP|','Share %']].copy()
            display_df3['Mean |SHAP|'] = display_df3['Mean |SHAP|'].round(4)
            display_df3['Share %']     = display_df3['Share %'].round(1).astype(str) + '%'
            st.dataframe(display_df3, use_container_width=True, hide_index=True)

        # Beeswarm
        st.markdown("#### SHAP Beeswarm Plot")
        shap.summary_plot(rq3_sv, rq3_X, show=False, plot_size=(10, 5))
        fig_bees3 = plt.gcf()
        fig_bees3.suptitle('RQ3 — SHAP Beeswarm (each dot = one hospital-DRG pair)',
                            fontsize=10, fontweight='bold', y=1.01)
        st.pyplot(fig_bees3); plt.close()

        st.markdown("- **DRG_Weight** accounts for **71.2%** of total model influence")
        st.markdown("- **BED_CNT** = 12.7% | DRG_Weight + BED_CNT together explain **84%**")
        st.markdown("- All RUCA geography groups combined = only **~1.6%**")

        # Subgroup error table
        st.markdown("#### Model Error by Subgroup (Test Set)")
        ec3, ec4 = st.columns(2)
        with ec3:
            st.markdown("**By Geography**")
            geo_err3 = pd.DataFrame({
                'RUCA Group': ['Metropolitan','Micropolitan','Small Town','Rural'],
                'MAE ($)':    [2315, 1163, 1177, 870],
                'MAPE (%)':   [14.6, 11.8, 11.5, 9.6],
                'Bias ($)':   [375, 139, 131, 129]
            })
            st.dataframe(geo_err3, use_container_width=True, hide_index=True)
        with ec4:
            st.markdown("**By Ownership**")
            own_err3 = pd.DataFrame({
                'Ownership':  ['For-Profit','Government','Non-Profit'],
                'MAE ($)':    [2340, 2102, 1609],
                'MAPE (%)':   [15.2, 12.6, 11.7],
                'Bias ($)':   [370, 412, 229]
            })
            st.dataframe(own_err3, use_container_width=True, hide_index=True)


# ============================================================
# TAB 4 — FORECAST
# ============================================================
with tab4:

    st.markdown('<div class="section-heading">2024 Medicare Forecast</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    2024 predictions are generated from <strong>synthetic inputs</strong> (2023 rows with Data_Year=2024),
    justified by CMS annual data release lag. RQ2 includes <strong>80% and 90% confidence intervals</strong>
    from XGBoost quantile regression — both calibrated to within 0.1% of target coverage on 2023 test set.
    </div>
    """, unsafe_allow_html=True)

    forecast_tab1, forecast_tab2 = st.tabs(["📊 RQ2 — Discharge Forecast", "💰 RQ3 — Reimbursement Forecast"])

    # ── RQ2 Forecast ──────────────────────────────────────
    with forecast_tab1:

        # Trend line: 2017–2023 + 2024
        st.markdown("#### National Discharge Trend — 2017–2023 Actual + 2024 Forecast")

        yearly_act = (df.groupby('Data_Year')['Tot_Dschrgs']
                        .mean().reset_index()
                        .rename(columns={'Tot_Dschrgs':'Mean_Dschrgs'}))
        mean_2024   = rq2_fc['Pred_Discharges'].mean()
        ci80_low    = rq2_fc['CI80_Low'].mean()
        ci80_high   = rq2_fc['CI80_High'].mean()
        ci90_low    = rq2_fc['CI90_Low'].mean()
        ci90_high   = rq2_fc['CI90_High'].mean()

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(yearly_act['Data_Year'], yearly_act['Mean_Dschrgs'],
                marker='o', color='#1565C0', linewidth=2.5, label='Actual (2017–2023)')
        ax.scatter([2024], [mean_2024], color='#EF6C00', s=100, zorder=5, label='2024 Forecast (median)')
        act_2023 = yearly_act[yearly_act['Data_Year']==2023]['Mean_Dschrgs'].values[0]
        ax.plot([2023, 2024], [act_2023, mean_2024], color='#EF6C00', linestyle='--', linewidth=1.5)
        ax.fill_between([2023.8, 2024.2], [ci90_low]*2, [ci90_high]*2,
                        alpha=0.15, color='#EF6C00', label='90% CI')
        ax.fill_between([2023.8, 2024.2], [ci80_low]*2, [ci80_high]*2,
                        alpha=0.35, color='#EF6C00', label='80% CI')
        ax.axvspan(2019.5, 2021.5, alpha=0.08, color='#C62828', label='COVID window')
        ax.set_xlabel('Year', fontsize=11); ax.set_ylabel('Mean Discharges per Hospital-DRG')
        ax.set_title('RQ2 — Discharge Volume Trend + 2024 Forecast', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        fc_col1, fc_col2 = st.columns(2)

        with fc_col1:
            # CI calibration metrics
            st.markdown("**CI Calibration on 2023 Test Set**")
            ci_cal = pd.DataFrame({
                'Interval': ['80% CI', '90% CI'],
                'Target Coverage': ['80%', '90%'],
                'Actual Coverage': ['80.1% ✅', '90.1% ✅']
            })
            st.dataframe(ci_cal, use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class="metric-row" style="margin-top:0.8rem">
                <div class="metric-card">
                    <div class="label">2024 Forecast Mean</div>
                    <div class="value">{mean_2024:.1f}</div>
                    <div class="sub">discharges per pair</div>
                </div>
                <div class="metric-card orange">
                    <div class="label">80% CI Range</div>
                    <div class="value">{ci80_low:.0f}–{ci80_high:.0f}</div>
                    <div class="sub">discharges</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with fc_col2:
            # Top 10 DRGs by predicted volume
            st.markdown("**Top 10 DRGs by Total 2024 Predicted Volume**")
            drg_vol = (rq2_fc.groupby('DRG_Cd')['Pred_Discharges']
                              .sum().sort_values(ascending=False).head(10)
                              .reset_index())
            drg_vol.columns = ['DRG Code', 'Total Predicted Discharges']
            st.dataframe(drg_vol, use_container_width=True, hide_index=True)

        # Download button
        st.markdown("---")
        csv_rq2 = rq2_fc.to_csv(index=False)
        st.download_button("⬇️ Download RQ2 2024 Predictions (CSV)",
                           data=csv_rq2,
                           file_name="RQ2_Predictions_2024_WithCI.csv",
                           mime="text/csv")

    # ── RQ3 Forecast ──────────────────────────────────────
    with forecast_tab2:

        st.markdown("#### Top 15 Highest Predicted Medicare Reimbursements (2024)")

        top15_rq3 = (rq3_fc.sort_values('Pred_Reimbursement', ascending=False).head(15))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        labels = top15_rq3['Rndrng_Prvdr_CCN'].astype(str) + ' | DRG ' + top15_rq3['DRG_Cd'].astype(str)
        bars = ax.barh(labels, top15_rq3['Pred_Reimbursement'],
                       color='#2E7D32', alpha=0.85)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
        ax.set_xlabel('Predicted Medicare Reimbursement')
        ax.set_title('Top 15 Hospital-DRG Pairs by Predicted 2024 Reimbursement',
                     fontsize=11, fontweight='bold')
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='x', alpha=0.25)
        for bar, v in zip(bars, top15_rq3['Pred_Reimbursement']):
            ax.text(v + 1000, bar.get_y()+bar.get_height()/2,
                    f'${v/1000:.0f}K', va='center', fontsize=8)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        fc_col3, fc_col4 = st.columns(2)

        with fc_col3:
            st.markdown(f"""
            <div class="metric-row" style="margin-top:0.5rem">
                <div class="metric-card green">
                    <div class="label">Mean Predicted Payment 2024</div>
                    <div class="value">${rq3_fc['Pred_Reimbursement'].mean():,.0f}</div>
                    <div class="sub">per hospital-DRG pair</div>
                </div>
                <div class="metric-card">
                    <div class="label">Max Predicted Payment</div>
                    <div class="value">${rq3_fc['Pred_Reimbursement'].max():,.0f}</div>
                    <div class="sub">high complexity DRGs</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with fc_col4:
            # Distribution of predicted payments
            fig, ax = plt.subplots(figsize=(5.5, 3))
            ax.hist(rq3_fc['Pred_Reimbursement'].clip(upper=100000),
                    bins=50, color='#2E7D32', alpha=0.8, edgecolor='white')
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
            ax.set_title('Distribution of 2024 Predicted Reimbursements', fontsize=9, fontweight='bold')
            ax.set_xlabel('Predicted Payment'); ax.set_ylabel('Count')
            ax.grid(axis='y', alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        st.markdown("---")
        csv_rq3 = rq3_fc.to_csv(index=False)
        st.download_button("⬇️ Download RQ3 2024 Predictions (CSV)",
                           data=csv_rq3,
                           file_name="RQ3_Predictions_2024.csv",
                           mime="text/csv")
