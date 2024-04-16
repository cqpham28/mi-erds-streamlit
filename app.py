import os
import streamlit as st
from src.pages import home, feedback

st.set_page_config(
    page_title="app",
    layout="wide",
)

PAGES = {
    "Home": home,
    "Feedback": feedback,
}



def main():

    selection = st.sidebar.radio("Select pages", 
                    list(PAGES.keys()))
    os.makedirs("refs", exist_ok=True)
    
    page = PAGES[selection]
    with st.spinner(f"Loading {selection} ..."):
        page.write()


if __name__ == "__main__":
    main()