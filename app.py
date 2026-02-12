import streamlit as st
import os

# ----------------------------------------------------
# Streamlit App Config
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Investment Advisor",
    layout="wide"
)

# ----------------------------------------------------
# APP TITLE
# ----------------------------------------------------
st.title("AI-Driven Financial Investment Advisor")
st.markdown(
    """
Welcome to your multi-agent powered investment system.  
Use the **left sidebar** to navigate through the analysis and recommendations.
    """
)

# ----------------------------------------------------
# AUTOMATICALLY DETECT AND DISPLAY PAGES
# ----------------------------------------------------
PAGES_DIR = "pages"

# If pages folder missing → warn user
if not os.path.exists(PAGES_DIR):
    st.error("Could not find `pages/` directory. Please create it.")
else:
    st.sidebar.title("Navigation")

    # Show Home link
    st.sidebar.page_link("app.py", label="Home")

    # Loop through pages folder
    for filename in sorted(os.listdir(PAGES_DIR)):
        if filename.endswith(".py"):
            page_path = f"{PAGES_DIR}/{filename}"
            page_label = filename.replace("_", " ").replace(".py", "")
            st.sidebar.page_link(page_path, label=f"{page_label}")

st.info("➡ Select an option from the sidebar to continue.")
