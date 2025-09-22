import streamlit as st
from ui.navbar import navbar
from pages import home, weights, compute_weights, scores, results, export, methodology

# Page configuration
st.set_page_config(
    page_title="AI Accreditation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    with open("styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Define pages
PAGES = {
    "home": home.render,
    "weights": weights.render,
    "compute_weights": compute_weights.render,
    "scores": scores.render,
    "results": results.render,
    "export": export.render,
    "methodology": methodology.render,
}

# Navigation
def main():
    # Get current page from session state or query params
    if "page" not in st.session_state:
        query_params = st.query_params
        if "page" in query_params and query_params["page"] in PAGES:
            st.session_state["page"] = query_params["page"]
        else:
            st.session_state["page"] = "home"
    
    # Render navbar and get selected page
    selected_page = navbar(list(PAGES.keys()), st.session_state["page"])
    
    # Update session state if page changed
    if selected_page != st.session_state["page"]:
        st.session_state["page"] = selected_page
        st.rerun()
    
    # Render the selected page
    if st.session_state["page"] in PAGES:
        PAGES[st.session_state["page"]]()
    else:
        st.error(f"Page '{st.session_state['page']}' not found")

if __name__ == "__main__":
    main()
