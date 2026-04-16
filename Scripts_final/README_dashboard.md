# Medicare Analytics Dashboard — Setup Guide
## DSCI 5260 | Group 6

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — File structure required

Place app.py inside your Scripts_final/ folder.
Make sure these paths exist relative to app.py:

```
Medicare DataAnalysis/
├── Scripts_final/
│   ├── app.py                  ← Streamlit app
│   ├── requirements.txt
│   └── outputs/
│       ├── rq2_shap_values.npy
│       ├── rq2_shap_base.npy
│       ├── rq2_shap_X.parquet
│       ├── rq3_shap_values.npy
│       ├── rq3_shap_base.npy
│       └── rq3_shap_X.parquet
└── Data/
    └── Processed_Data/
        ├── df_medidata_clean.parquet
        ├── rq2_xgb_model.pkl
        ├── rq3_xgb_model.pkl
        ├── hosp_te_lookup.pkl
        ├── drg_te_lookup.pkl
        ├── RQ2_Predictions_2024_WithCI.csv
        └── RQ3_Predictions_2024.csv
```

---

## Step 3 — Run locally

```bash
cd Medicare DataAnalysis/Scripts_final
streamlit run app.py
```

Opens at: http://localhost:8501

---

## Step 4 — Set up Google Drive (optional)

1. Upload all files from Data/Processed_Data/ to a Google Drive folder
2. Right-click each file → Share → Anyone with the link → Viewer
3. Copy the file ID from each link:
   https://drive.google.com/file/d/FILE_ID_HERE/view
4. Open app.py and replace values in the GDRIVE_IDS dictionary:
   'data':      'paste_file_id_here',
   'rq2_model': 'paste_file_id_here',
   ... etc
5. In the app sidebar, toggle ON "Load from Google Drive"

---

## Step 5 — Share with others (free hosting)

1. Push app.py and requirements.txt to a GitHub repo
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Select your repo → app.py → Deploy
5. Share the public URL with your professor and team

Note: For large files (models, parquet), use Google Drive loading
rather than uploading to GitHub (100MB file limit).

---

## Tab Guide

| Tab | What it shows |
|-----|--------------|
| EDA | Trends, distributions, ownership/geography comparisons |
| Predict | Interactive RQ2 + RQ3 prediction with any hospital/DRG |
| SHAP | Feature importance bar charts + beeswarm plots |
| Forecast | 2024 discharge + reimbursement forecasts with CI bands |

---

## Adding New Charts to EDA Tab

Open app.py and find the comment:
    # ── Row 4: Top DRGs by payment gap

Add your chart code below any existing section.
Example:

    st.markdown('<div class="section-heading">My New Chart</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    # your matplotlib code here
    st.pyplot(fig)
    plt.close()

Save the file. The app refreshes automatically.
