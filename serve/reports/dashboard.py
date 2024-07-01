import numpy as np
import plotly.figure_factory as ff
import streamlit as st

st.markdown("# Dashboard")

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]

group_labels = ["Group 1", "Group 2", "Group 3"]

# Create distplot with custom bin_size
fig = ff.create_distplot(
    hist_data, group_labels, bin_size=[.1, .25, .5]
)

# Plot!
st.plotly_chart(fig, use_container_width=True)
