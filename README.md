# 🎯 AI Accreditation Dashboard – Week 3

This project is part of the **AI Accreditation study**.  
The goal is to build a **Streamlit dashboard** that evaluates operators against ethical criteria using:

- **Gatekeepers** (minimum threshold checks)
- **ARAS** (Additive Ratio Assessment System utility calculation)
- **Tiering** (Platinum / Gold / Silver / Bronze / Not Accredited based on ARAS score)

👉 Live demo: [AI Accreditation – Week 3](https://ai-accreditation.streamlit.app/)

---

## 📂 Project Structure

```
ai-accreditation/
├─ streamlit_app.py       
├─ helpers.py              
├─ pages/                  
│  ├─ 1_Weights.py              # Week-1 view of Delphi + weights
│  ├─ 2_Compute_Weights_W2.py   # Week-2 weight computation
│  ├─ 2_Scores.py               # Criteria + operator scores
│  ├─ 3_Results.py              # Week-3: Gatekeepers, ARAS, Tiers
│  ├─ 4_Export.py               # Snapshot export
├─ data/                   
│  ├─ criteria_catalog.csv
│  ├─ delphi_round1_example.csv
│  ├─ weights_from_delphi_example.csv
│  ├─ operator_scores_dummy.csv
└─ requirements.txt        
```

---

## 🚀 Features (Week-3)

- ✅ **Multi-page Streamlit app** with navigation + topbar  
- ✅ **CSV loading & previews** for criteria, Delphi inputs, weights, operator scores  
- ✅ **Delphi weight computation** (mean / median / trimmed mean, Week-2)  
- ✅ **Gatekeeper evaluation** with pass/fail and reasons per operator  
- ✅ **ARAS utility (K)** calculation with proper specification compliance
- ✅ **Tier assignment** (Platinum / Gold / Silver / Bronze / Not Accredited)  
- ✅ **Debug expanders** to inspect weights, normalization, and decision matrix  
- ✅ **Charts**: operator utilities and tier distribution  
- ✅ **CSV Export snapshot**  

---

## 🛠 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/valetivivek/ai-accreditation.git
   cd ai-accreditation
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📊 Pages Overview

- **Home (`streamlit_app.py`)**  
  Quick project overview + navigation links.

- **Weights (`pages/1_Weights.py`)**  
  Shows Delphi expert ratings and any precomputed weights.

- **Compute Weights (W2) (`pages/2_Compute_Weights_W2.py`)**  
  Aggregates Delphi ratings into normalized weights.

- **Scores (`pages/2_Scores.py`)**  
  Displays the criteria catalog and raw operator scores.

- **Results (W3) (`pages/3_Results.py`)**  
  Gatekeeper pass/fail, ARAS utility K values, and tier assignment.

- **Export (`pages/4_Export.py`)**  
  Allows downloading a snapshot CSV of loaded tables.

---

## 📦 Data Requirements

Place the following CSVs inside the `/data` directory:

- `criteria_catalog.csv` (includes criterion_id, type, gate_min/gate_max)
- `delphi_round1_example.csv` (expert ratings 1–9)
- `weights_from_delphi_example.csv` (optional precomputed weights; else use Compute Weights page)
- `operator_scores_dummy.csv` (operator_id, criterion_id, score)

---
