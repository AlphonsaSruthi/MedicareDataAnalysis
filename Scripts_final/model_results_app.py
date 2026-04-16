# ============================================================
# Medicare Analytics — Model Results App
# DSCI 5260 — Group 6
# No data files needed — all results are hardcoded
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Medicare Model Results | Group 6",
    page_icon="📈",
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
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
    background: #f0f4f9; border-radius: 10px; padding: 1rem 1.4rem;
    flex: 1; min-width: 140px; border-top: 3px solid #2196F3;
}
.metric-card.green  { border-top-color: #2E7D32; }
.metric-card.orange { border-top-color: #EF6C00; }
.metric-card .label { font-size: 0.95rem; color: #607080; text-transform: uppercase; font-weight: 700; margin-bottom: 0.4rem; }
.metric-card .value { font-size: 1.9rem; font-weight: 700; color: #0d1f35; font-family: 'IBM Plex Mono', monospace; }
.metric-card .sub   { font-size: 0.85rem; color: #607080; margin-top: 0.25rem; }
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

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Results")
    st.markdown("---")
    st.markdown("**No data files needed**")
    st.markdown("All results are pre-computed from the modeling notebooks.")
    st.markdown("---")
    st.markdown("**Models Compared**")
    st.markdown("- Linear Regression (baseline)\n- Random Forest\n- XGBoost (selected)")
    st.markdown("---")
    st.caption("DSCI 5260 · Group 6")

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <h1>📈 Model Comparison — All 3 Models Tried</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
Both RQ2 and RQ3 tested three models: Linear Regression (baseline), Random Forest (intermediate),
and XGBoost (final best model). The <strong style="background:#d4edda;padding:1px 4px;border-radius:3px;">green highlighted row</strong>
is the XGBoost Test set result — the final honest evaluation on unseen data.
</div>
""", unsafe_allow_html=True)

# ── Helper ──────────────────────────────────────────────────
def df_to_html_green(df, green_model, green_split):
    cols = df.columns.tolist()
    header = "<tr>" + "".join(
        f"<th style='background:#f0f4f9;font-weight:700;padding:6px 10px;text-align:left;border-bottom:2px solid #ddd;'>{c}</th>"
        for c in cols) + "</tr>"
    rows = ""
    for _, row in df.iterrows():
        is_green = (row['Model'] == green_model and row['Split'] == green_split)
        bg = "background-color:#d4edda;font-weight:bold;" if is_green else ""
        cells = "".join(
            f"<td style='padding:6px 10px;border-bottom:1px solid #eee;{bg}'>{row[c]}</td>"
            for c in cols)
        rows += f"<tr>{cells}</tr>"
    return f"<table style='width:100%;border-collapse:collapse;font-size:0.88rem;'><thead>{header}</thead><tbody>{rows}</tbody></table>"

# ── Tabs ────────────────────────────────────────────────────
rq2_tab, rq3_tab = st.tabs(["  RQ2 — Discharge Volume  ", "  RQ3 — Medicare Reimbursement  "])

# ── RQ2 ─────────────────────────────────────────────────────
with rq2_tab:
    st.markdown('<div class="section-heading">RQ2 — Predicting Medicare Discharge Volume</div>', unsafe_allow_html=True)
    st.markdown("**Target:** `Tot_Dschrgs` (log-transformed) | **Split:** Temporal — Train 2017–2021, Val 2022, Test 2023")

    st.markdown("#### Final Test Set Result — XGBoost")
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card green">
            <div class="label">R² (log scale)</div>
            <div class="value">0.682</div>
            <div class="sub">68.2% variance explained</div>
        </div>
        <div class="metric-card green">
            <div class="label">R² (count scale)</div>
            <div class="value">0.581</div>
            <div class="sub">58.1% on real discharge counts</div>
        </div>
        <div class="metric-card">
            <div class="label">MAE (discharges)</div>
            <div class="value">12</div>
            <div class="sub">Average prediction error</div>
        </div>
        <div class="metric-card orange">
            <div class="label">MAPE</div>
            <div class="value">31.3%</div>
            <div class="sub">Relative error per prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Full Results — Train / Validation / Test")
    rq2_full = pd.DataFrame({
        'Model': ['Linear Regression','Linear Regression','Random Forest','Random Forest','XGBoost','XGBoost','XGBoost'],
        'Split': ['Train','Val','Train','Val','Train','Val','Test'],
        'RMSE':  [0.5445, 0.5349, 0.4045, 0.4157, 0.3273, 0.3547, 0.3804],
        'MAE (log)': [0.4264, 0.4098, 0.3168, 0.3249, 0.2529, 0.2736, 0.2914],
        'MAE (discharges)': [62, 66, 13, 13, 10, 10, 12],
        'MAPE':  ['79.1%','92.9%','34.7%','36.8%','27.4%','30.4%','31.3%'],
        'R²':    [0.407, 0.387, 0.673, 0.630, 0.786, 0.731, 0.682],
        'R² (count)': ['-','-', 0.649, 0.629, 0.836, 0.799, 0.581],
    })
    st.markdown(df_to_html_green(rq2_full, 'XGBoost', 'Test'), unsafe_allow_html=True)

    st.markdown("#### Model Performance Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    models = ['Linear\nRegression', 'Random\nForest', 'XGBoost']
    colors = ['#C62828', '#EF6C00', '#1565C0']

    bars = axes[0].bar(models, [0.387, 0.630, 0.731], color=colors, width=0.5, edgecolor='white')
    axes[0].set_title('R² — Validation Set (higher = better)', fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, 1); axes[0].spines[['top','right']].set_visible(False)
    axes[0].axhline(0.7, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    for bar, v in zip(bars, [0.387, 0.630, 0.731]):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    bars2 = axes[1].bar(models, [66, 13, 10], color=colors, width=0.5, edgecolor='white')
    axes[1].set_title('MAE — Validation Set (discharges, lower = better)', fontsize=10, fontweight='bold')
    axes[1].spines[['top','right']].set_visible(False)
    for bar, v in zip(bars2, [66, 13, 10]):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+0.3, f'{v}', ha='center', fontsize=10, fontweight='bold')

    fig.suptitle('RQ2 — Model Comparison', fontsize=11, fontweight='bold')
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Why XGBoost Was Selected")
    st.markdown('''<div class="key-insight-box">
    <strong>Best R² (0.731)</strong> and lowest MAE (10 discharges) on validation set.
    Train/Val gap of 0.055 is slightly above 0.05 but acceptable given the COVID structural break in 2020–2021.
    Linear Regression failed on the count scale (negative R²) confirming non-linear relationships.
    </div>''', unsafe_allow_html=True)

    st.markdown("#### Why Linear Regression Failed")
    st.dataframe(pd.DataFrame({
        'Metric': ['R² (log scale)', 'R² (count scale)', 'MAE (discharges)', 'MAPE'],
        'Train':  ['0.407', '-32,300', '62', '79.1%'],
        'Val':    ['0.387', '-39,835', '66', '92.9%'],
        'Reason': [
            'Captures only 39% of variance — non-linear patterns missed',
            'Negative R² — worse than predicting the mean on real counts',
            'Off by 66 discharges on average — unacceptable for planning',
            '93% error — completely unreliable for discharge forecasting'
        ]
    }), use_container_width=True, hide_index=True)

# ── RQ3 ─────────────────────────────────────────────────────
with rq3_tab:
    st.markdown('<div class="section-heading">RQ3 — Predicting Medicare Reimbursement</div>', unsafe_allow_html=True)
    st.markdown("**Target:** `Avg_Mdcr_Pymt_Amt` (log-transformed) | **Split:** Random — 70% Train, 15% Val, 15% Test")

    st.markdown("#### Final Test Set Result — XGBoost")
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card green">
            <div class="label">R² (log scale)</div>
            <div class="value">0.9216</div>
            <div class="sub">92.2% variance explained</div>
        </div>
        <div class="metric-card green">
            <div class="label">R² ($ scale)</div>
            <div class="value">0.8511</div>
            <div class="sub">85.1% on real dollar amounts</div>
        </div>
        <div class="metric-card">
            <div class="label">MAE ($)</div>
            <div class="value">$2,190</div>
            <div class="sub">Average prediction error</div>
        </div>
        <div class="metric-card orange">
            <div class="label">MAPE</div>
            <div class="value">14.3%</div>
            <div class="sub">Relative error per prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Full Results — Train / Validation / Test")
    rq3_full = pd.DataFrame({
        'Model': ['Linear Regression','Linear Regression','Random Forest','Random Forest','XGBoost','XGBoost','XGBoost'],
        'Split': ['Train','Val','Train','Val','Train','Val','Test'],
        'RMSE':  [0.407, 0.410, 0.243, 0.244, 0.191, 0.199, 0.199],
        'MAE (log)': [0.297, 0.298, 0.182, 0.183, 0.138, 0.144, 0.144],
        'MAE ($)': ['$82,448','$115,344','$2,790','$2,827','$2,069','$2,205','$2,190'],
        'MAPE':  ['57.1%','66.9%','17.9%','18.0%','13.7%','14.3%','14.3%'],
        'R²':    [0.673, 0.669, 0.884, 0.882, 0.928, 0.922, 0.922],
        'R² ($)': ['-','-', 0.736, 0.732, 0.879, 0.850, 0.851],
    })
    st.markdown(df_to_html_green(rq3_full, 'XGBoost', 'Test'), unsafe_allow_html=True)

    st.markdown("#### Model Performance Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors_r3 = ['#C62828', '#EF6C00', '#1565C0']

    bars3 = axes[0].bar(['Linear\nRegression','Random\nForest','XGBoost'],
                        [0.669, 0.882, 0.922], color=colors_r3, width=0.5, edgecolor='white')
    axes[0].set_title('R² — Validation Set (higher = better)', fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, 1); axes[0].spines[['top','right']].set_visible(False)
    axes[0].axhline(0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    for bar, v in zip(bars3, [0.669, 0.882, 0.922]):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    bars4 = axes[1].bar(['Linear\nRegression','Random\nForest','XGBoost'],
                        [115344, 2827, 2205], color=colors_r3, width=0.5, edgecolor='white')
    axes[1].set_title('MAE ($) — Validation Set (lower = better)', fontsize=10, fontweight='bold')
    axes[1].spines[['top','right']].set_visible(False)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
    for bar, v in zip(bars4, [115344, 2827, 2205]):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+500, f'${v/1000:.0f}K', ha='center', fontsize=10, fontweight='bold')

    fig.suptitle('RQ3 — Model Comparison', fontsize=11, fontweight='bold')
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Why XGBoost Was Selected")
    st.markdown('''<div class="key-insight-box">
    <strong>Best R² (0.922)</strong> and lowest MAE ($2,205) on validation set.
    Near-zero Train/Val gap (0.001) confirms no overfitting. Linear Regression produced
    a <strong>negative R² on the dollar scale</strong> — confirming log-scale OLS is unsuitable
    for Medicare payment prediction.
    </div>''', unsafe_allow_html=True)

    st.markdown("#### Why Linear Regression Failed")
    st.dataframe(pd.DataFrame({
        'Metric': ['R² (log scale)', 'R² ($ scale)', 'MAE ($)', 'MAPE'],
        'Train':  ['0.673', 'Negative', '$82,448', '57.1%'],
        'Val':    ['0.669', '-273,793', '$115,344', '66.9%'],
        'Reason': [
            'Looks decent in log scale — misleading',
            'Negative on dollar scale — worse than predicting the mean',
            'Off by $115K per discharge — operationally useless',
            '67% average error — cannot be used for payment forecasting'
        ]
    }), use_container_width=True, hide_index=True)
