import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# LOADING DATA
DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)

@st.cache(persist=True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
    return data

data = load_data(100000)

# CREATING FUNCTION FOR MAPS

def map(data, lat, lon, zoom):
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": lat,
            "longitude": lon,
            "zoom": zoom,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["lon", "lat"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ]
    ))

# LAYING OUT THE TOP SECTION OF THE APP
t0,t1, t2, t3, t4 = st.beta_columns((1,8,1,12,1))

t1.title("NYC Ridesharing Data")

hour_selected = t1.slider("Select hour of pickup", 0, 23)

t3.write(
    """
    ##
    Examining how Uber pickups vary over time in New York City's and at its major regional airports.
    By sliding the slider on the left you can view different slices of time and explore different transportation trends.
    """)

# FILTERING DATA BY HOUR SELECTED
data = data[data[DATE_TIME].dt.hour == hour_selected]

# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
c0, c1, c2, c3, c4, c5, c6 = st.beta_columns((1,8,1,4,4,4,1))

# SETTING THE ZOOM LOCATIONS FOR THE AIRPORTS
la_guardia= [40.7900, -73.8700]
jfk = [40.6650, -73.7821]
newark = [40.7090, -74.1805]
zoom_level = 12
midpoint = (np.average(data["lat"]), np.average(data["lon"]))

with c1:
    st.write("**All New York City from %i:00 and %i:00**" % (hour_selected, (hour_selected + 1) % 24))
    map(data, midpoint[0], midpoint[1], 11)

with c3:
    st.write("**La Guardia Airport from %i:00 and %i:00**" % (hour_selected, (hour_selected + 1) % 24))
    map(data, la_guardia[0],la_guardia[1], zoom_level)

with c4:
    st.write("**JFK Airport from %i:00 and %i:00**" % (hour_selected, (hour_selected + 1) % 24))
    map(data, jfk[0],jfk[1], zoom_level)

with c5:
    st.write("**Newark Airport from %i:00 and %i:00**" % (hour_selected, (hour_selected + 1) % 24))
    map(data, newark[0],newark[1], zoom_level)

# FILTERING DATA FOR THE HISTOGRAM
filtered = data[
    (data[DATE_TIME].dt.hour >= hour_selected) & (data[DATE_TIME].dt.hour < (hour_selected + 1))
    ]

hist = np.histogram(filtered[DATE_TIME].dt.minute, bins=60, range=(0, 60))[0]

chart_data = pd.DataFrame({"minute": range(60), "pickups": hist})

# LAYING OUT THE HISTOGRAM SECTION
z0, z1, z2 = st.beta_columns((1,21,1))

z1.write("")

z1.write("**Breakdown by minute between %i:00 and %i:00**" % (hour_selected, (hour_selected + 1) % 24))

z1.altair_chart(alt.Chart(chart_data)
    .mark_area(
        interpolate='step-after',
    ).encode(
        x=alt.X("minute:Q", scale=alt.Scale(nice=False)),
        y=alt.Y("pickups:Q"),
        tooltip=['minute', 'pickups']
    ).configure_mark(
        opacity=0.5,
        color='red'
    ), use_container_width=True)