
# 🎯 AI Accreditation Dashboard – Week 1

This project is part of the **AI Accreditation study**.  
The goal is to build a **Streamlit dashboard** that evaluates operators against ethical criteria using:

- **Gatekeepers** (minimum checks)
- **ARAS** (Additive Ratio Assessment System)
- **Tiering** (Platinum / Gold / Silver / Bronze)

👉 Live demo: [AI Accreditation – Week 1](https://ai-accreditation.streamlit.app/)

---

## 📂 Project Structure

```
ai-accreditation/
├─ streamlit_app.py       
├─ helpers.py              
├─ pages/                  
│  ├─ 1_Weights.py          
│  ├─ 2_Scores.py         
│  ├─ 3_Results.py         
│  ├─ 4_Export.py          
├─ data/                   
│  ├─ criteria_catalog.csv
│  ├─ delphi_round1_example.csv
│  ├─ weights_from_delphi_example.csv
│  ├─ operator_scores_dummy.csv
└─ requirements.txt        
```

---

## 🚀 Features (Week-1)

- ✅ **Multi-page Streamlit app** with navigation
- ✅ **Topbar with Home button**
- ✅ **Loads & previews CSVs** (criteria, Delphi inputs, weights, operator scores)
- ✅ **Data availability check**
- ✅ **CSV Export snapshot**
- 🔜 Future: Gatekeepers, ARAS utility, tiering system

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
  Shows Delphi expert ratings and optional precomputed weights.

- **Scores (`pages/2_Scores.py`)**  
  Displays the criteria catalog and raw operator scores.

- **Results (`pages/3_Results.py`)**  
  Placeholder for Gatekeepers, ARAS, and tiering logic (coming weeks).

- **Export (`pages/4_Export.py`)**  
  Allows downloading a snapshot CSV of loaded tables.

---

## 📦 Data Requirements

Place the following CSVs inside the `/data` directory:

- `criteria_catalog.csv`
- `delphi_round1_example.csv`
- `weights_from_delphi_example.csv`
- `operator_scores_dummy.csv`

---
