import streamlit as st
from ase.data import chemical_symbols
from pymatgen.core import Element

elements = [Element.from_Z(z) for z in range(1, 119)]

# Define the number of rows and columns in the periodic table
rows = 9  # There are 7 rows in the conventional periodic table
columns = 18

# Define a function to display the periodic table
def display_periodic_table():
    # elements = [
    #     (element, element) for element in chemical_symbols[1:]
    # ]

    # cols = st.columns(18, gap='small', vertical_alignment='bottom')  # Create 18 columns for the periodic table layout

    row = 0
    for element in elements:
        symbol = element.symbol
        atomic_number = element.Z
        group = element.group

        if element.row > row:
            cols = st.columns(columns, gap='small', vertical_alignment='bottom')
        row = element.row

        if element.block == 'f':
            continue

        with cols[group - 1]:
            if st.button(symbol, use_container_width=True):
                st.session_state.selected_element = symbol
                st.session_state.selected_name = symbol
                st.rerun()
                # st.experimental_rerun()
    
    for element in elements:
        symbol = element.symbol
        atomic_number = element.Z
        group = element.group

        if element.row > row:
            cols = st.columns(columns, gap='small', vertical_alignment='bottom')
        row = element.row

        if element.block == 'f':
            noble = Element.from_row_and_group(row-1, 18)
            row += 2
            group += atomic_number - noble.Z - 2
        else:
            continue

        with cols[group - 1]:
            if st.button(symbol, use_container_width=True):
                st.session_state.selected_element = symbol
                st.session_state.selected_name = symbol
                st.rerun()
                # st.experimental_rerun()
    
    
    # for idx, (symbol, name) in enumerate(elements):
    #     with cols[idx % 18]:  # Place each element in the correct column
    #         if st.button(symbol, use_container_width=True):
    #             st.session_state.selected_element = symbol
    #             st.session_state.selected_name = name
    #             st.experimental_rerun()

# Define a function to display the details of an element
def display_element_details():
    symbol = st.session_state.selected_element
    name = st.session_state.selected_name
    st.write(f"### {name} ({symbol})")
    st.write(f"Details about {name} ({symbol}) will be displayed here.")
    if st.button("Back to Periodic Table"):
        st.session_state.selected_element = None
        st.session_state.selected_name = None
        st.experimental_rerun()


st.title("Periodic Table")

# st.balloons()
if 'selected_element' not in st.session_state:
    st.session_state.selected_element = None

if st.session_state.selected_element:
    display_element_details()
else:
    display_periodic_table()

