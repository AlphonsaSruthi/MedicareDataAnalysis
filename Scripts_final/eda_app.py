# ============================================================
# Medicare Inpatient Analytics — EDA Dashboard
# DSCI 5260 — Group 6
# Lightweight EDA-only app
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Medicare EDA | Group 6",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

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
    margin: 0;
    letter-spacing: -0.5px;
}

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

.section-heading {
    font-size: 1.25rem;
    font-weight: 800;
    color: #1565C0;
    border-bottom: 2.5px solid #1565C0;
    padding-bottom: 0.45rem;
    margin: 1.6rem 0 1rem 0;
    letter-spacing: 0.2px;
}

.key-insight-box {
    background: #e8f4fd;
    border-left: 4px solid #1565C0;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    font-size: 0.95rem;
    color: #0d1f35;
    margin: 0.8rem 0;
    font-weight: 500;
}

.info-box {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    font-size: 0.85rem;
    color: #4a3700;
    margin: 0.8rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a1628 !important;
    min-width: 280px !important;
}
section[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
section[data-testid="stSidebar"] hr { border-color: #1e3a5f !important; }

/* Expander headers */
details summary p,
[data-testid="stExpander"] summary p {
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    color: #1565C0 !important;
    letter-spacing: 0.2px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stExpander"] {
    border: none !important;
    border-bottom: 2.5px solid #1565C0 !important;
    border-radius: 0 !important;
    margin: 1.6rem 0 1rem 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE CONFIGURATION
# ============================================================

GDRIVE_IDS = {
    'data': '17_miiDElUZx6XdjOY25h94BPT8qsPGlj',  # df_medidata_clean.parquet
}

LOCAL_PATHS = {
    'data': '../Data/Processed_Data/df_medidata_clean.parquet',
}

def gdrive_url(file_id):
    return f"https://drive.google.com/uc?id={file_id}"

def load_file(key, use_gdrive=False):
    if use_gdrive:
        try:
            import gdown
            local = f"/tmp/{key}_cached.parquet"
            if not os.path.exists(local):
                gdown.download(gdrive_url(GDRIVE_IDS[key]), local, quiet=True)
            return local
        except Exception as e:
            st.warning(f"Google Drive load failed: {e}. Falling back to local.")
    return LOCAL_PATHS[key]

@st.cache_data(show_spinner=False)
def load_data(use_gdrive):
    path = load_file('data', use_gdrive)

    REQUIRED_COLS = [
        'Rndrng_Prvdr_CCN', 'Rndrng_Prvdr_State_Abrvtn',
        'Rndrng_Prvdr_RUCA', 'DRG_Cd', 'DRG_Desc', 'DRG_Weight',
        'Tot_Dschrgs', 'Avg_Submtd_Cvrd_Chrg', 'Avg_Mdcr_Pymt_Amt',
        'Data_Year', 'Ownership_Type', 'BED_CNT',
        'own_For-Profit', 'own_Non-Profit', 'own_Government',
        'ruca_Metropolitan', 'ruca_Micropolitan', 'ruca_Small Town', 'ruca_Rural',
        'outlier_payment_flag',
    ]

    import pyarrow.parquet as pq
    available = pq.read_schema(path).names
    cols_to_load = [c for c in REQUIRED_COLS if c in available]
    df = pd.read_parquet(path, columns=cols_to_load)

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

    # Downcast to save memory
    for col in df.select_dtypes(include='float64').columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include='int64').columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in ['Ownership_Type', 'RUCA_Group', 'Rndrng_Prvdr_State_Abrvtn']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Medicare EDA")
    st.markdown("---")

    # Auto-detect cloud vs local
    is_cloud = not os.path.exists('../Data/Processed_Data/df_medidata_clean.parquet')
    use_gdrive = st.toggle("☁️ Load from Google Drive", value=is_cloud,
                           help="ON = Google Drive (cloud). OFF = local files.")

    st.markdown("---")
    st.markdown("**Data Source**")
    st.markdown("CMS Medicare Inpatient  \n2017–2023 | ~1.18M records")
    st.markdown("---")
    st.caption("DSCI 5260 · Group 6")


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-bar">
    <h1>Medicare Inpatient Analytics — Exploratory Data Analysis</h1>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================
with st.spinner("Loading data..."):
    try:
        df = load_data(use_gdrive)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()


# ============================================================
# EDA CONTENT
# ============================================================

# ── Dataset Overview KPI Cards ─────────────────────────────
st.markdown('<div class="section-heading">Dataset Overview</div>', unsafe_allow_html=True)

total_records   = len(df)
total_hospitals = df['Rndrng_Prvdr_CCN'].nunique()
total_drgs      = df['DRG_Cd'].nunique()
median_gap      = df['Payment_Gap'].median()
median_ratio    = df['Payment_Ratio'].median()

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


# ── National Trends ────────────────────────────────────────
st.markdown('<div class="section-heading">National Trends (2017–2023)</div>', unsafe_allow_html=True)

yearly = (df.groupby('Data_Year')
            .agg(Mean_Dschrgs = ('Tot_Dschrgs', 'mean'),
                 Median_Gap   = ('Payment_Gap',  'median'))
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
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("- COVID dip clearly visible in 2020–2021")
    st.markdown("- Gradual national decline in discharge volume post-2021")

with col2:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(yearly['Data_Year'], yearly['Median_Gap'],
            marker='s', color='#C62828', linewidth=2.2, markersize=6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
    ax.set_title('Median Payment Gap Over Time', fontsize=11, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Median Gap ($)')
    ax.grid(True, alpha=0.25); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("- Payment gap widening over time")
    st.markdown("- Billed charges rise faster than Medicare payments")


# ── Payment Gap by Hospital Type ──────────────────────────
st.markdown('<div class="section-heading">Payment Gap by Hospital Type</div>', unsafe_allow_html=True)

OWN_PAL  = {'For-Profit': '#1565C0', 'Non-Profit': '#EF6C00', 'Government': '#2E7D32'}
RUCA_PAL = {'Metropolitan': '#1565C0', 'Micropolitan': '#EF6C00', 'Small Town': '#F9A825', 'Rural': '#2E7D32'}
RUCA_ORD = ['Metropolitan', 'Micropolitan', 'Small Town', 'Rural']

col3, col4 = st.columns(2)
with col3:
    own_df  = df[df['Ownership_Type'].isin(['For-Profit','Non-Profit','Government'])]
    metrics = own_df.groupby('Ownership_Type').agg(
        Payment_Ratio            = ('Payment_Ratio',             'median'),
        payment_gap_per_discharge= ('payment_gap_per_discharge', 'median'),
    ).reindex(['For-Profit','Non-Profit','Government'])

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
    for ax, col_name, title, fmt in zip(
        axes,
        ['Payment_Ratio', 'payment_gap_per_discharge'],
        ['Payment Ratio\n(Medicare / Billed)', 'Gap per Discharge ($)'],
        ['{:.2f}', '${:,.0f}']
    ):
        vals = metrics[col_name]
        bars = ax.bar(vals.index, vals.values,
                      color=[OWN_PAL[o] for o in vals.index], width=0.5, edgecolor='white')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=15, labelsize=8)
        ax.spines[['top','right']].set_visible(False)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                    fmt.format(v), ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    fig.suptitle('Ownership Type Comparison', fontsize=10, fontweight='bold')
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("- Non-Profit carries the largest gap per discharge: **$2,480**")
    st.markdown("- Non-Profit recovers only **13¢ per $1 billed**")

with col4:
    geo_df      = df[df['RUCA_Group'].isin(RUCA_ORD)]
    geo_metrics = geo_df.groupby('RUCA_Group').agg(
        Payment_Ratio            = ('Payment_Ratio',             'median'),
        payment_gap_per_discharge= ('payment_gap_per_discharge', 'median'),
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
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("- Rural hospitals recover **36¢ per $1 billed** vs Metropolitan **19¢**")
    st.markdown("- Metropolitan carries the **largest payment gap** per discharge")


# ── Metric Definitions ────────────────────────────────────
st.markdown('<div class="section-heading">Understanding the Metrics</div>', unsafe_allow_html=True)

def_col1, def_col2 = st.columns(2)
with def_col1:
    st.markdown("""
    <div style="background:#fff3e0;border-left:4px solid #EF6C00;border-radius:0 8px 8px 0;padding:1rem 1.2rem;">
        <div style="font-size:1rem;font-weight:800;color:#E65100;margin-bottom:0.5rem;">Payment Gap = Billed &minus; Paid</div>
        <div style="font-size:0.88rem;color:#4a3700;">
            The <strong>dollar amount</strong> the hospital never receives from Medicare.<br><br>
            <strong>Example:</strong> $100,000 billed &rarr; $20,000 paid &rarr; Gap = <strong>$80,000</strong><br><br>
            Tells you: <em>How much money is missing?</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

with def_col2:
    st.markdown("""
    <div style="background:#e8f5e9;border-left:4px solid #2E7D32;border-radius:0 8px 8px 0;padding:1rem 1.2rem;">
        <div style="font-size:1rem;font-weight:800;color:#1B5E20;margin-bottom:0.5rem;">Payment Ratio = Paid &divide; Billed</div>
        <div style="font-size:0.88rem;color:#1a3700;">
            The <strong>fraction of the bill</strong> Medicare actually covers.<br><br>
            <strong>Example:</strong> $20,000 &divide; $100,000 = <strong>0.20</strong> (20 cents per $1 billed)<br><br>
            Tells you: <em>How fair is Medicare's coverage?</em>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Ownership × Geography Heatmap ─────────────────────────
st.markdown('<div class="section-heading">Ownership × Geography Interaction</div>', unsafe_allow_html=True)

own_geo_df = df[
    df['Ownership_Type'].isin(['For-Profit','Non-Profit','Government']) &
    df['RUCA_Group'].isin(['Metropolitan','Micropolitan','Small Town','Rural'])
].copy()

pivot_gap = own_geo_df.pivot_table(
    values='payment_gap_per_discharge', index='Ownership_Type', columns='RUCA_Group', aggfunc='median'
).reindex(index=['For-Profit','Government','Non-Profit'],
          columns=['Metropolitan','Micropolitan','Small Town','Rural'])

pivot_ratio = own_geo_df.pivot_table(
    values='Payment_Ratio', index='Ownership_Type', columns='RUCA_Group', aggfunc='median'
).reindex(index=['For-Profit','Government','Non-Profit'],
          columns=['Metropolitan','Micropolitan','Small Town','Rural'])

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
im1 = axes[0].imshow(pivot_gap.values, cmap='YlOrRd', aspect='auto')
axes[0].set_xticks(range(4)); axes[0].set_xticklabels(['Metropolitan','Micropolitan','Small Town','Rural'], fontsize=9)
axes[0].set_yticks(range(3)); axes[0].set_yticklabels(['For-Profit','Government','Non-Profit'], fontsize=9)
axes[0].set_title('Median Payment Gap ($000s)\n(dark = larger gap)', fontsize=10, fontweight='bold')
for i in range(3):
    for j in range(4):
        v = pivot_gap.values[i,j]
        if not np.isnan(v):
            axes[0].text(j, i, f'${v/1000:.0f}K', ha='center', va='center', fontsize=9, fontweight='bold',
                         color='white' if v > pivot_gap.values.max()*0.6 else 'black')
plt.colorbar(im1, ax=axes[0]).set_label('Payment Gap ($)', fontsize=8)

im2 = axes[1].imshow(pivot_ratio.values, cmap='RdYlGn', aspect='auto', vmin=0.05, vmax=0.50)
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(['Metropolitan','Micropolitan','Small Town','Rural'], fontsize=9)
axes[1].set_yticks(range(3)); axes[1].set_yticklabels(['For-Profit','Government','Non-Profit'], fontsize=9)
axes[1].set_title('Median Payment Ratio\n(green = Medicare pays higher share)', fontsize=10, fontweight='bold')
for i in range(3):
    for j in range(4):
        v = pivot_ratio.values[i,j]
        if not np.isnan(v):
            axes[1].text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=9, fontweight='bold',
                         color='white' if v < 0.15 else 'black')
plt.colorbar(im2, ax=axes[1]).set_label('Payment Ratio', fontsize=8)

fig.suptitle('Ownership x Geography — Payment Gap and Payment Ratio Heatmaps', fontsize=11, fontweight='bold')
fig.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("- **Non-Profit Metropolitan** hospitals have the worst combination — largest gap AND lowest ratio")
st.markdown("- **Government Rural** hospitals recover the highest share — best ratio across all groups")


# ── OLS Scatter — Expander ────────────────────────────────
with st.expander("DRG Severity vs Payment Gap — OLS Trend"):
    st.markdown(
        "**X-axis:** DRG complexity weight &nbsp;|&nbsp; "
        "**Y-axis:** Dollar gap (billed − paid) &nbsp;|&nbsp; "
        "**Color:** Payment ratio — red = severe underpayment, green = better coverage"
    )
    from scipy import stats as scipy_stats

    drg_scatter = (df.groupby('DRG_Cd')
                     .agg(median_ratio  = ('Payment_Ratio', 'median'),
                          median_weight = ('DRG_Weight',    'median'),
                          median_gap    = ('Payment_Gap',   'median'))
                     .reset_index().dropna())

    slope_gap, intercept_gap, r_gap, _, _ = scipy_stats.linregress(
        drg_scatter['median_weight'], drg_scatter['median_gap'])
    x_line = np.linspace(drg_scatter['median_weight'].min(), drg_scatter['median_weight'].max(), 100)

    fig, ax = plt.subplots(figsize=(13, 6))
    sc = ax.scatter(drg_scatter['median_weight'], drg_scatter['median_gap'],
                    c=drg_scatter['median_ratio'], cmap='RdYlGn', s=45, alpha=0.85,
                    edgecolors='none', vmin=0.05, vmax=0.50)
    ax.plot(x_line, slope_gap * x_line + intercept_gap, 'k--', linewidth=2,
            label=f'OLS trend  (slope = ${slope_gap:,.0f} per unit weight)')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Payment Ratio', fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.2f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
    ax.set_xlabel('Median DRG Severity Weight', fontsize=11)
    ax.set_ylabel('Median Payment Gap ($000s)', fontsize=11)
    ax.set_title(f'DRG Weight vs Payment Gap  (r = {r_gap:.2f})\n'
                 f'Each dot = one DRG  |  Colour = payment ratio  |  Higher weight → larger dollar gap',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False); ax.grid(alpha=0.2)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="key-insight-box"><strong>Key Insight:</strong> Higher complexity = '
                'Medicare pays more dollars, but the gap keeps growing because '
                '<strong>billed charges grow even faster</strong>.</div>', unsafe_allow_html=True)


# ── Top 20 DRGs ───────────────────────────────────────────
st.markdown('<div class="section-heading">Top 20 DRGs by Median Payment Gap</div>', unsafe_allow_html=True)

drg_gap = (df.groupby(['DRG_Cd','DRG_Desc'])
             .agg(median_gap   = ('Payment_Gap',   'median'),
                  median_ratio = ('Payment_Ratio', 'median'))
             .reset_index()
             .sort_values('median_gap', ascending=False)
             .head(20))
drg_gap['Label'] = drg_gap['DRG_Cd'].astype(str) + ': ' + drg_gap['DRG_Desc'].str[:55]

def drg_color(ratio):
    if ratio < 0.15:   return '#C62828'
    elif ratio < 0.25: return '#EF6C00'
    else:              return '#1565C0'

colors = [drg_color(r) for r in drg_gap['median_ratio']]

fig, ax = plt.subplots(figsize=(13, 7))
bars = ax.barh(drg_gap['Label'], drg_gap['median_gap'], color=colors, alpha=0.9)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
ax.set_xlabel('Median Payment Gap ($000s)', fontsize=10)
ax.set_title('Top 20 DRGs by Median Payment Gap\n(payment gap = billed charge − Medicare payment)',
             fontsize=11, fontweight='bold')
ax.spines[['top','right']].set_visible(False); ax.grid(axis='x', alpha=0.25)
for bar, v, r in zip(bars, drg_gap['median_gap'], drg_gap['median_ratio']):
    ax.text(v + 8000, bar.get_y()+bar.get_height()/2,
            f'ratio={r:.2f}', va='center', fontsize=7.5, color='#333333')
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#C62828', label='Payment ratio < 0.15  (severe underpayment)'),
    Patch(facecolor='#EF6C00', label='Payment ratio 0.15–0.25'),
    Patch(facecolor='#1565C0', label='Payment ratio > 0.25'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
fig.tight_layout(); st.pyplot(fig); plt.close()


# ── Summary Stats + Expanders ─────────────────────────────
with st.expander("📋 Summary Statistics Table"):
    key_cols = ['Tot_Dschrgs','Avg_Submtd_Cvrd_Chrg','Avg_Mdcr_Pymt_Amt','Payment_Gap','Payment_Ratio']
    summary = df[key_cols].describe().round(2)
    summary = summary.rename(index={'50%':'median'})
    st.dataframe(summary, use_container_width=True)

with st.expander("📈 Payment Gap Trend by Ownership Type (2017–2023)"):
    OWN_PALETTE_EXP = {'For-Profit': '#1565C0', 'Non-Profit': '#EF6C00', 'Government': '#2E7D32'}
    df_own_exp = df[df['Ownership_Type'].isin(['For-Profit','Non-Profit','Government'])].copy()
    yr_own = (df_own_exp.groupby(['Data_Year','Ownership_Type'])
                        .agg(median_gap   = ('Payment_Gap',   'median'),
                             median_ratio = ('Payment_Ratio', 'median'))
                        .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Payment Gap Trend by Ownership Type (2017–2023)', fontsize=12, fontweight='bold')
    for own_type, grp in yr_own.groupby('Ownership_Type'):
        col_c = OWN_PALETTE_EXP[own_type]
        axes[0].plot(grp['Data_Year'], grp['median_gap']/1000, 'o-', color=col_c,
                     linewidth=2, markersize=6, label=own_type)
        axes[1].plot(grp['Data_Year'], grp['median_ratio'], 'o-', color=col_c,
                     linewidth=2, markersize=6, label=own_type)
    for ax, title, ylabel in zip(
        axes,
        ['Median Payment Gap by Ownership × Year', 'Payment Ratio by Ownership × Year'],
        ['Median Payment Gap ($000s)', 'Median Payment Ratio']
    ):
        ax.set_title(title, fontsize=10, fontweight='bold'); ax.set_ylabel(ylabel)
        ax.set_xticks(yr_own['Data_Year'].unique())
        ax.axvline(2020, color='grey', linestyle=':', linewidth=1, label='COVID 2020')
        ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.spines[['top','right']].set_visible(False)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}K'))
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("- **Non-Profit gap consistently highest** across all years and widening post-2020")
    st.markdown("- **Non-Profit ratio consistently lowest** — recovers the smallest fraction every year")

with st.expander("🗺️ State-Level Payment Gap — Top 10 Highest vs Lowest"):
    state_agg = (df.groupby('Rndrng_Prvdr_State_Abrvtn')
                   .agg(median_gap   = ('Payment_Gap',   'median'),
                        median_ratio = ('Payment_Ratio', 'median'))
                   .reset_index()
                   .rename(columns={'Rndrng_Prvdr_State_Abrvtn': 'State'})
                   .sort_values('median_gap', ascending=False))
    top10 = state_agg.head(10)
    bot10 = state_agg.tail(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('State-Level Payment Gap — Top 10 Highest vs Lowest', fontsize=12, fontweight='bold')
    axes[0].barh(top10['State'][::-1], top10['median_gap'][::-1]/1000, color='#C62828', edgecolor='white', alpha=0.9)
    for i, (_, row) in enumerate(top10[::-1].iterrows()):
        axes[0].text(row['median_gap']/1000 + 0.3, i, f"ratio={row['median_ratio']:.2f}", va='center', fontsize=8.5)
    axes[0].set_title('Top 10 States — Largest Gap', fontsize=10, fontweight='bold')
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}K'))
    axes[0].grid(axis='x', alpha=0.3); axes[0].spines[['top','right']].set_visible(False)

    axes[1].barh(bot10['State'], bot10['median_gap']/1000, color='#2E7D32', edgecolor='white', alpha=0.9)
    for i, (_, row) in enumerate(bot10.iterrows()):
        axes[1].text(row['median_gap']/1000 + 0.2, i, f"ratio={row['median_ratio']:.2f}", va='center', fontsize=8.5)
    axes[1].set_title('Bottom 10 States — Smallest Gap', fontsize=10, fontweight='bold')
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}K'))
    axes[1].grid(axis='x', alpha=0.3); axes[1].spines[['top','right']].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("- States with the largest gaps tend to have **lower payment ratios**")
    st.markdown("- Geographic variation reflects differences in billing practices and DRG mix")


# ── Distribution Analysis ─────────────────────────────────
st.markdown('<div class="section-heading">Distribution Analysis — Log Transformation</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    axes[0].hist(df['Tot_Dschrgs'].clip(upper=500), bins=50, color='#1565C0', alpha=0.8, edgecolor='white')
    axes[0].set_title('Discharge Volume (Raw)', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Discharges'); axes[0].set_ylabel('Count')
    axes[1].hist(np.log1p(df['Tot_Dschrgs']), bins=50, color='#EF6C00', alpha=0.8, edgecolor='white')
    axes[1].set_title('Discharge Volume (Log)', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('log1p(Discharges)'); axes[1].set_ylabel('Count')
    fig.suptitle('RQ2 Target — Before & After Log Transform', fontsize=9, fontweight='bold')
    fig.tight_layout(); st.pyplot(fig); plt.close()

with col6:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    axes[0].hist(df['Avg_Mdcr_Pymt_Amt'].clip(upper=100000), bins=50, color='#2E7D32', alpha=0.8, edgecolor='white')
    axes[0].set_title('Medicare Payment (Raw)', fontsize=9, fontweight='bold')
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
    axes[0].set_ylabel('Count')
    axes[1].hist(np.log1p(df['Avg_Mdcr_Pymt_Amt']), bins=50, color='#00838F', alpha=0.8, edgecolor='white')
    axes[1].set_title('Medicare Payment (Log)', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('log1p($)'); axes[1].set_ylabel('Count')
    fig.suptitle('RQ3 Target — Before & After Log Transform', fontsize=9, fontweight='bold')
    fig.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("- All financial variables are **right-skewed** — log transformation normalises distributions")
st.markdown("- Back-transformed with `expm1()` for predictions in modeling notebooks")
