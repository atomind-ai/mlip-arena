import streamlit as st

# Assuming you have similar modules as in the example
# For demonstration, these functions will be defined here directly
def home_page():
    st.title("Home Page")
    st.write("Welcome to the Home Page!")

def analytics_page():
    st.title("Analytics Page")
    st.write("Analytics details go here.")

def settings_page():
    st.title("Settings Page")
    st.write("Settings details go here.")

# Mimicking the page_group utility from the example
class PageGroup:
    def __init__(self, key):
        self.key = key
        self.pages = {}
        self.default_page = None

    def item(self, title, func, default=False):
        self.pages[title] = func
        if default:
            self.default_page = title

    def show(self):
        # Use session state to remember the current page
        if 'current_page' not in st.session_state or st.session_state.current_page not in self.pages:
            st.session_state.current_page = self.default_page

        # Display the current page function
        self.pages[st.session_state.current_page]()

def main():
    page = PageGroup("p")

    with st.sidebar:
        st.title("Navigation")

        with st.expander("Pages", True):
            for title, func in [("Home", home_page), ("Analytics", analytics_page), ("Settings", settings_page)]:
                
                if st.button(title):
                    st.session_state.current_page = title

    page.item("Home", home_page, default=True)
    page.item("Analytics", analytics_page)
    page.item("Settings", settings_page)

    page.show()

if __name__ == "__main__":
    # st.set_page_config(page_title="My Streamlit App", page_icon="📊", layout="wide")
    st.set_page_config(
        layout="wide",
        page_title="MLIP Arena",
        page_icon=":shark:",
        # initial_sidebar_state="expanded",
        menu_items=None
    )
    main()