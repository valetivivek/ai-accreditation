# 🎯 AI Accreditation Dashboard – Week 1

This project is part of the AI Accreditation study. The goal is to build a **Streamlit dashboard** that evaluates operators against ethical criteria using Gatekeepers, ARAS (Additive Ratio Assessment System), and tiering (Platinum/Gold/Silver/Bronze).

👉 Live demo (Week-1): [https://ai-accreditation-week-1-deliverables.streamlit.app/](https://ai-accreditation-week-1-deliverables.streamlit.app/)

---

## 📂 Project Structure
```
ai-accreditation/
├─ streamlit_app.py        # Streamlit app entry point
├─ requirements.txt        # Python dependencies
├─ data/                   # Input data files
│  ├─ criteria_catalog.csv
│  ├─ delphi_round1_example.csv
│  ├─ weights_from_delphi_example.csv
│  └─ operator_scores_dummy.csv
└─ README.md
```

---

## 🚀 Running Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/valetivivek/ai-accreditation.git
   cd ai-accreditation
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   streamlit run streamlit_app.py
   ```
4. Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Week-1 Milestone
- ✅ Created project structure with `/data` subfolder.  
- ✅ Added CSV files and confirmed they load.  
- ✅ Built static dashboard pages:
  - **Overview** – shows project goals.  
  - **Weights** – displays Delphi expert inputs & (if provided) precomputed weights.  
  - **Scores** – shows operator scores and criteria catalog.  
  - **Results** – placeholder (to be filled in from Week-3 onward).  
  - **Export** – allows downloading a snapshot CSV of current data.  