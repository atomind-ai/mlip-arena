import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path("mlip_arena/tasks/diatomics")
methods = ["MACE-MP", "Equiformer", "CHGNet", "MACE-OFF"]
dfs = [pd.read_json(DATA_DIR / method.lower() /  "homonuclear-diatomics.json") for method in methods]
df = pd.concat(dfs, ignore_index=True)

table = pd.DataFrame(columns=["Model", "No. of supported elements", "No. of reversed forces", "Energy-consistent forces"])

for method in df["method"].unique():
    rows = df[df["method"] == method]
    new_row = {
        "Model": method,
        "No. of supported elements": len(rows["name"].unique()),
        "No. of reversed forces": None,  # Replace with actual logic if available
        "Energy-consistent forces": None  # Replace with actual logic if available
    }
    table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)



# Define the data
# data = {
#     "Metrics": [
#         "No. of supported elements",
#         "No. of reversed forces",
#         "Energy-consistent forces",
#     ],
#     "MACE-MP(M)": ["10", "5", "Yes"],
#     "CHGNet": ["20", "3", "No"],
#     "Equiformer": ["15", "7", "Yes"]
# }

# # Convert the data to a DataFrame
# df = pd.DataFrame(data)

# # Set the 'Metrics' column as the index
# df.set_index("Metrics", inplace=True)

# # Transpose the DataFrame
# df = df.T

# Apply custom CSS to center the table
# Create the Streamlit table

table.set_index("Model", inplace=True)


s = table.style.background_gradient(
    cmap="Spectral", 
    subset=["No. of supported elements"],
    vmin=0, vmax=120
)


st.markdown("# Leaderboard")
st.dataframe(s, use_container_width=True)

# Define custom CSS for table
# custom_css = """
# <style>
# table {
#     width: 100%;
#     border-collapse: collapse;
# }
# th, td {
#     border: 1px solid #ddd;
#     padding: 8px;
# }
# th {
#     background-color: #4CAF50;
#     color: white;
#     text-align: left;
# }
# tr:nth-child(even) {
#     background-color: #f2f2f2;
# }
# tr:hover {
#     background-color: #ddd;
# }
# </style>
# """

# # Display the table with custom CSS
# st.markdown(custom_css, unsafe_allow_html=True)
# st.markdown(table.to_html(index=False), unsafe_allow_html=True)





# import numpy as np
# import plotly.figure_factory as ff
# import streamlit as st

# st.markdown("# Dashboard")

# # Add histogram data
# x1 = np.random.randn(200) - 2
# x2 = np.random.randn(200)
# x3 = np.random.randn(200) + 2

# # Group data together
# hist_data = [x1, x2, x3]

# group_labels = ["Group 1", "Group 2", "Group 3"]

# # Create distplot with custom bin_size
# fig = ff.create_distplot(
#     hist_data, group_labels, bin_size=[.1, .25, .5]
# )

# # Plot!
# st.plotly_chart(fig, use_container_width=True)
