import os
import streamlit as st
from src.pages import home, feedback, benchmark
import hmac

st.set_page_config(
    page_title="app",
    layout="wide",
)

#---------------#
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()

#--------------------------#
PAGES = {
    "Home": home,
    "Feedback": feedback,
    "Benchmark": benchmark,
}

##-------PARAMETER------##
if "s3_df" not in st.session_state: 
    st.session_state.s3_df = []

if "s3_img" not in st.session_state: 
    st.session_state.s3_img = []

if "current_run" not in st.session_state: 
    st.session_state.current_run = 0

if "current_file" not in st.session_state: 
    st.session_state.current_file = ""

if "path_edf" not in st.session_state: 
    st.session_state.path_edf = {}

if "data_run" not in st.session_state: 
    st.session_state.data_run = {
        "hand": {
            "event_ids": dict(left_hand=1, 
                            right_hand=2),
            },

        "foot": {
            "event_ids": dict(left_foot=4, 
                            right_foot=3),
            },
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