

# NOTE: https://stackoverflow.com/questions/77062368/streamlit-bokeh-event-callback-to-get-clicked-values
# Taptool: https://docs.bokeh.org/en/2.4.2/docs/reference/models/tools.html#taptool

import streamlit as st
from bokeh.plotting import figure
from bokeh.plotting import figure, show
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap

import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, TapTool
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap


periods = ["I", "II", "III", "IV", "V", "VI", "VII"]
groups = [str(x) for x in range(1, 19)]

df = elements.copy()
df["atomic mass"] = df["atomic mass"].astype(str)
df["group"] = df["group"].astype(str)
df["period"] = [periods[x-1] for x in df.period]
df = df[df.group != "-"]
df = df[df.symbol != "Lr"]
df = df[df.symbol != "Lu"]

cmap = {
    "alkali metal"         : "#a6cee3",
    "alkaline earth metal" : "#1f78b4",
    "metal"                : "#d93b43",
    "halogen"              : "#999d9a",
    "metalloid"            : "#e08d49",
    "noble gas"            : "#eaeaea",
    "nonmetal"             : "#f1d4Af",
    "transition metal"     : "#599d7A",
}

TOOLTIPS = [
    ("Name", "@name"),
    ("Atomic number", "@{atomic number}"),
    ("Atomic mass", "@{atomic mass}"),
    ("Type", "@metal"),
    ("CPK color", "$color[hex, swatch]:CPK"),
    ("Electronic configuration", "@{electronic configuration}"),
]

p = figure(title="Periodic Table (omitting LA and AC Series)", width=1000, height=450, 
           x_range=groups, y_range=list(reversed(periods)), 
           tools="hover,tap", toolbar_location=None, tooltips=TOOLTIPS)

# Convert DataFrame to ColumnDataSource
df["selected"] = False
source = ColumnDataSource(df)

r = p.rect("group", "period", 0.95, 0.95, source=source, fill_alpha=0.6, 
           legend_field="metal", 
           color=factor_cmap('metal', palette=list(cmap.values()), factors=list(cmap.keys())),
           selection_color="firebrick", selection_alpha=0.9)


# r = p.rect("group", "period", 0.95, 0.95, source=df, fill_alpha=0.6, legend_field="metal",
#            color=factor_cmap('metal', palette=list(cmap.values()), factors=list(cmap.keys())))

text_props = dict(source=df, text_align="left", text_baseline="middle")

x = dodge("group", -0.4, range=p.x_range)

p.text(x=x, y="period", text="symbol", text_font_style="bold", **text_props)

p.text(x=x, y=dodge("period", 0.3, range=p.y_range), text="atomic number",
       text_font_size="11px", **text_props)

p.text(x=x, y=dodge("period", -0.35, range=p.y_range), text="name",
       text_font_size="7px", **text_props)

p.text(x=x, y=dodge("period", -0.2, range=p.y_range), text="atomic mass",
       text_font_size="7px", **text_props)

p.text(x=["3", "3"], y=["VI", "VII"], text=["LA", "AC"], text_align="center", text_baseline="middle")

p.outline_line_color = None
p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_standoff = 0
p.legend.orientation = "horizontal"
p.legend.location ="top_center"
p.hover.renderers = [r] # only hover element boxes

print(source.dataspecs())

# Create a CustomJS callback
callback = CustomJS(args=dict(source=source), code="""
        var data = source.data;
        var selected_elements = [];
        for (var i = 0; i < data.symbol.length; i++) {
            if (data.selected[i]) { // Corrected if statement with braces
                selected_elements.push(data.symbol[i]);
            }
        }
        console.log('Selected elements:', selected_elements);
        document.dispatchEvent(new CustomEvent("selection_event", {detail: JSON.stringify(selected_elements)}));
    """)
    # yield j
    # st.rerun()
    
    

# Add TapTool with the callback
tap_tool = TapTool()
p.add_tools(tap_tool)
p.js_on_event('tap', callback)

st.bokeh_chart(p, use_container_width=True)

# show(p)

selected_info = st.empty()

# Use session state to store selected elements
if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = []

st.markdown("""
<script>
document.addEventListener('selection_event', function(e) {
    var selected_elements = JSON.parse(e.detail);
    window.parent.postMessage({
        type: 'streamlit:set_session_state',
        data: {
            selected_elements: selected_elements
        }
    }, '*');
});
</script>
""", unsafe_allow_html=True)

# Display selected elements
if st.session_state.selected_elements:
    st.write("Selected Elements:")
    for element in st.session_state.selected_elements:
        st.write(f"{element['symbol']} ({element['name']}):")
        st.write(f"  Atomic Number: {element['atomic_number']}")
        st.write(f"  Atomic Mass: {element['atomic_mass']}")
        st.write(f"  Type: {element['metal']}")
        st.write("---")
        
else:
    st.write("No elements selected. Click on elements in the periodic table to select them.")
    # st.rerun()

# Add a button to clear selection
if st.button("Clear Selection"):
    st.session_state.selected_elements = []
    st.rerun()
