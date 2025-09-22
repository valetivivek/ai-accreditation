# This is basically app bootstrap and global UI
from helpers import topbar

topbar()
import streamlit as st

st.markdown("""
**Week-3 goals** ✅ **COMPLETED**
- ✅ Confirm project structure and `/data` files exist  
- ✅ Load and display CSVs on dedicated pages  
- ✅ Compute Delphi weights (Week-2)  
- ✅ Implement Gatekeeper pass/fail logic  
- ✅ Calculate ARAS utility (K) with proper specification compliance
- ✅ Assign tiers (Platinum / Gold / Silver / Bronze / Not Accredited)
- ✅ Enhanced debugging and numerical safety
""")


st.divider()
st.subheader("Quick Navigation")
st.page_link("pages/1_Weights.py", label="➡️ Weights")
st.page_link("pages/2_Scores.py",  label="➡️ Scores")
st.page_link("pages/3_Results.py", label="➡️ Results")
st.page_link("pages/4_Export.py",  label="➡️ Export")
st.page_link("pages/2_Compute_Weights_W2.py", label="➡️ Compute Weights (W2)")
