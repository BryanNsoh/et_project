"""
OpenET Irrigation Advisory - Streamlit App (Refined Alignment & Address Search)
"""
import os
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw, MeasureControl
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from dotenv import load_dotenv
import requests

# Load local modules (assume these exist in your project)
from openet_api import fetch_openet_data, OpenETError, load_api_key
from data_cache import load_cache, save_cache
from irrigation import get_irrigation_recommendation
from utils import validate_geometry, calculate_area, format_coordinates_for_display

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App configuration
APP_TITLE = "OpenET Irrigation Advisory"
# Default center near 402 West State Farm Road, North Platte, NE
DEFAULT_CENTER = [41.121092, -100.768147]
DEFAULT_ZOOM = 6
MAX_AREA_ACRES = 50000

# Attempt to load API keys from environment
OPENET_API_KEY = os.environ.get('OPENET_API_KEY', '')
GEOAPIFY_API_KEY = os.environ.get('GEOAPIFY_API_KEY', '')

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS to style the app ---
st.markdown("""
<style>
/* Simplify spacing around main container */
.main .block-container {
    padding-top: 1rem;
}

/* Headings */
h1, h2, h3 {
    color: #1e5b94;
}
h2 {
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 0.5rem;
}

/* Button styling */
.stButton>button {
    background-color: #1e5b94;
    color: white;
    border: none;
    border-radius: 4px;
}
.stButton>button:hover {
    background-color: #164576;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #173d5e;
}
[data-testid="stSidebar"] .st-bq,
[data-testid="stSidebar"] .st-c0,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSubheader {
    color: white !important;
}

/* Instructions box */
.instructions-box {
    background-color: #173d5e;
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.instructions-box ol {
    padding-left: 1.5rem;
    margin-bottom: 0;
}
.instructions-box li {
    margin-bottom: 0.5rem;
    color: white;
}

/* Status boxes */
.status-box {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.info-box { background-color: #f0f6fb; border-left: 5px solid #4287f5; }
.success-box { background-color: #f0f9f5; border-left: 5px solid #13ab5c; }
.warning-box { background-color: #fffaf0; border-left: 5px solid #f7b034; }
.error-box { background-color: #fef2f1; border-left: 5px solid #ef564d; }

/* White background for code blocks, etc. */
pre code {
    background-color: #f8f8f8;
    color: #333;
}

/* Slight top padding for horizontal alignment */
section.block-container {
    padding-top: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)
# --- END CUSTOM CSS ---

def initialize_session_state():
    """Initialize Streamlit session state variables if they don't exist."""
    today = datetime.now().date()
    last_year = today - timedelta(days=365)

    if 'user_polygon' not in st.session_state:
        st.session_state['user_polygon'] = None

    if 'geometry_metadata' not in st.session_state:
        st.session_state['geometry_metadata'] = None

    if 'et_data' not in st.session_state:
        st.session_state['et_data'] = None

    if 'last_query_params' not in st.session_state:
        st.session_state['last_query_params'] = None

    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = OPENET_API_KEY  # pre-load if found

    # Default date values
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = last_year.isoformat()
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = today.isoformat()

    # Default map center & zoom
    if 'map_center' not in st.session_state:
        st.session_state['map_center'] = DEFAULT_CENTER
    if 'map_zoom' not in st.session_state:
        st.session_state['map_zoom'] = DEFAULT_ZOOM

@st.cache_data(ttl=3600*24)
def get_et_data_cached(
    geometry: List,
    start_date: str,
    end_date: str,
    interval: str = "daily",
    model: str = "Ensemble",
    variable: str = "ET",
    units: str = "mm"
) -> pd.DataFrame:
    """
    Cached function to get ET data from OpenET API or local cache.
    """
    df = load_cache(
        coords=geometry,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        model=model,
        variable=variable,
        max_age_hours=24
    )
    if df is not None:
        return df

    # If not in local cache, call OpenET API
    try:
        df = fetch_openet_data(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            model=model,
            variable=variable,
            units=units
        )
        save_cache(
            coords=geometry,
            start_date=start_date,
            end_date=end_date,
            data=df,
            interval=interval,
            model=model,
            variable=variable
        )
        return df

    except OpenETError as e:
        st.cache_data.clear()
        raise e

def create_map(center=None, zoom=None):
    """
    Create an interactive map with drawing tools + ESRI World Imagery.
    """
    if center is None:
        center = st.session_state['map_center']
    if zoom is None:
        zoom = st.session_state['map_zoom']

    m = folium.Map(location=center, zoom_start=zoom, control_scale=True)

    # Add ESRI World Imagery basemap
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri, Maxar, Earthstar Geographics, and the GIS User Community',
        name='ESRI World Imagery',
        overlay=False,
        control=True
    ).add_to(m)

    # Add drawing tools
    draw = Draw(
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={
            'poly': {'allowIntersection': False},
            'featureGroup': None
        }
    )
    draw.add_to(m)

    # Add measurement tool
    measure = MeasureControl(
        position='topleft',
        primary_length_unit='kilometers',
        secondary_length_unit='miles',
        primary_area_unit='hectares',
        secondary_area_unit='acres'
    )
    measure.add_to(m)

    # If a polygon is already selected, show it on the map
    if st.session_state['user_polygon'] is not None:
        coords = st.session_state['user_polygon']
        # Assume polygon
        if not isinstance(coords[0], (int, float)):
            folium.Polygon(
                locations=coords,
                color='#1e5b94',
                fill=True,
                fill_color='#1e5b94',
                fill_opacity=0.3,
                tooltip='Selected Field'
            ).add_to(m)

            # Zoom to the polygon
            sw_lat = min(lat for _, lat in coords)
            sw_lon = min(lon for lon, _ in coords)
            ne_lat = max(lat for _, lat in coords)
            ne_lon = max(lon for lon, _ in coords)
            m.fit_bounds([[sw_lat, sw_lon], [ne_lat, ne_lon]])

    return m

def render_map_section():
    """
    Field Selection: 
    1) Address search with multi-match selectbox
    2) Folium map for drawing fields
    3) Clear selection button on the side
    """
    st.header("Field Selection")

    # Hide text label for alignment; place label text in a markdown just above
    st.markdown("**Search by Address**")
    addr_col, btn_col = st.columns([4, 1])
    with addr_col:
        # No label, to keep alignment next to button
        address_query = st.text_input(
            label="",
            placeholder="Type an address, e.g. '402 West State Farm Road, North Platte, NE'",
            key="address_query_input",
            label_visibility="collapsed"
        )
    with btn_col:
        search_clicked = st.button("Search", key="address_search_button")

    # The list of candidate addresses is stored in session state for multiple tries
    if 'address_candidates' not in st.session_state:
        st.session_state['address_candidates'] = []

    if search_clicked and address_query and GEOAPIFY_API_KEY:
        # Query Geoapify for possible matches
        geo_url = f"https://api.geoapify.com/v1/geocode/autocomplete?text={address_query}&apiKey={GEOAPIFY_API_KEY}"
        try:
            resp = requests.get(geo_url, timeout=10)
            resp.raise_for_status()
            features = resp.json().get("features", [])

            if not features:
                st.warning("No address matches found.")
                st.session_state['address_candidates'] = []
            else:
                # Store possible matches in session
                st.session_state['address_candidates'] = features
        except requests.exceptions.RequestException as e:
            st.error(f"Address lookup failed: {e}")
    elif search_clicked and not GEOAPIFY_API_KEY:
        st.warning("No GEOAPIFY_API_KEY found. Cannot perform address search.")

    # If we have candidates, let user pick from them
    if st.session_state['address_candidates']:
        suggestions = [feat["properties"]["formatted"] for feat in st.session_state['address_candidates']]
        selected_addr = st.selectbox(
            "Select the correct match:",
            options=suggestions,
            key="address_match_selectbox"
        )

        locate_clicked = st.button("Locate Address", key="locate_address_button")
        if locate_clicked and selected_addr:
            chosen = None
            for feat in st.session_state['address_candidates']:
                if feat["properties"]["formatted"] == selected_addr:
                    chosen = feat
                    break
            if chosen:
                lat = chosen["properties"]["lat"]
                lon = chosen["properties"]["lon"]
                # Update map center & zoom
                st.session_state['map_center'] = [lat, lon]
                st.session_state['map_zoom'] = 18
                # Clear the suggestions after use
                st.session_state['address_candidates'] = []
                st.experimental_rerun()

    # Columns: map (left) & instructions/clear (right)
    map_col, control_col = st.columns([4,1])
    with map_col:
        # Create and display the map
        m = create_map()
        map_data = st_folium(m, height=450, width="100%")

        # Process user drawings
        if map_data and map_data.get("all_drawings"):
            drawings = map_data["all_drawings"]
            if drawings:
                last_drawing = drawings[-1]
                geometry = last_drawing.get("geometry", {})
                geom_type = geometry.get("type", "")

                if geom_type in ["Polygon", "Rectangle"]:
                    coords = geometry.get("coordinates", [[]])[0]
                    try:
                        validate_geometry(coords, "polygon")
                        area_sq_m = calculate_area(coords)
                        area_acres = area_sq_m * 0.000247105

                        if area_acres <= MAX_AREA_ACRES:
                            st.session_state['user_polygon'] = coords
                            st.session_state['geometry_metadata'] = {
                                "geometry_type": "polygon",
                                "area_acres": area_acres,
                                "area_sq_m": area_sq_m
                            }
                        else:
                            st.error(f"Area exceeds limit of {MAX_AREA_ACRES} acres.")
                    except ValueError as e:
                        st.error(str(e))

    with control_col:
        # Instructions
        st.markdown("""
        <div class="instructions-box" style="font-size:0.9rem;">
            <b>How to Select a Field:</b>
            <ol>
                <li>Use the rectangle ⬜ or polygon 🔺 tools.</li>
                <li>Draw your boundary.</li>
                <li>The field will highlight upon drawing.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # Clear selection button (only if a polygon is currently stored)
        if st.session_state['user_polygon']:
            if st.button("Clear Selection", key="clear_selection"):
                st.session_state['user_polygon'] = None
                st.session_state['geometry_metadata'] = None
                st.experimental_rerun()

def fetch_data(params):
    """Fetch OpenET data and store in session state."""
    if st.session_state['user_polygon'] is None:
        st.warning("Please select or draw a field on the map first.")
        return

    if not st.session_state['api_key']:
        st.warning("An API key is required to fetch data from OpenET. Please enter it in the sidebar.")
        return

    start_date = params['start_date']
    end_date = params['end_date']
    current_params = (
        str(st.session_state['user_polygon']),
        start_date.isoformat(),
        end_date.isoformat(),
        params['interval'],
        params['model'],
        params['variable'],
        params['units']
    )

    # If these parameters match the last fetch, skip re-fetch
    if st.session_state['last_query_params'] == current_params and st.session_state['et_data'] is not None:
        st.success("Using previously fetched data. (Parameters unchanged.)")
        return

    with st.spinner("Retrieving data from OpenET..."):
        try:
            today = datetime.now().date()
            # Ensure not to exceed today's date
            end_str = min(end_date, today).isoformat()

            df = get_et_data_cached(
                geometry=st.session_state['user_polygon'],
                start_date=start_date.isoformat(),
                end_date=end_str,
                interval=params['interval'],
                model=params['model'],
                variable=params['variable'],
                units=params['units']
            )

            if df is None or df.empty:
                st.error("No data returned. This may be due to coverage limitations or date range issues.")
            else:
                st.session_state['et_data'] = df
                st.session_state['last_query_params'] = current_params
                st.success(f"Data fetched: {len(df)} records from {start_date} to {end_str}")

        except OpenETError as e:
            st.error(f"Error fetching data: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

def render_sidebar():
    """Render the sidebar with additional app settings or info."""
    st.sidebar.title("OpenET Irrigation Advisory")

    # API key input in the sidebar
    with st.sidebar.expander("API Key Settings", expanded=False):
        api_key = st.text_input(
            "OpenET API Key",
            value=st.session_state.get('api_key', ''),
            type="password",
            help="Required to fetch data from OpenET."
        )
        st.session_state['api_key'] = api_key
        if not api_key:
            st.warning("No API key entered; you cannot fetch data from OpenET without it.")

    # Time period selection
    st.sidebar.subheader("Time Period")
    def update_start_date():
        st.session_state['start_date'] = st.session_state['start_date_input'].isoformat()
    def update_end_date():
        st.session_state['end_date'] = st.session_state['end_date_input'].isoformat()

    today = datetime.now().date()
    future_date = today + timedelta(days=365*10)
    default_start = datetime.fromisoformat(st.session_state['start_date']).date()
    default_end = datetime.fromisoformat(st.session_state['end_date']).date()

    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        min_value=datetime(1985, 1, 1).date(),
        max_value=future_date,
        key="start_date_input",
        on_change=update_start_date
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end,
        min_value=datetime(1985, 1, 1).date(),
        max_value=future_date,
        key="end_date_input",
        on_change=update_end_date
    )

    # Ensure valid range
    if end_date < start_date:
        st.sidebar.error("End date cannot be before start date")

    # Data interval, model, variable, and units
    st.sidebar.subheader("Data Parameters")
    interval = st.sidebar.selectbox("Data Interval", ["daily", "monthly"], index=0)
    model = st.sidebar.selectbox("ET Model",
                                 ["Ensemble", "SSEBop", "SIMS", "PTJPL", "eeMETRIC", "DisALEXI"],
                                 index=0)
    variable = st.sidebar.selectbox("Variable",
                                    ["ET", "ETo", "ETof", "NDVI", "PR"],
                                    index=0)
    units = st.sidebar.selectbox("Units", ["mm", "in"], index=0)

    return {
        'start_date': start_date,
        'end_date': end_date,
        'interval': interval,
        'model': model,
        'variable': variable,
        'units': units
    }

def render_results(params):
    """If data is fetched, show results & irrigation analysis."""
    if st.session_state['et_data'] is None:
        st.info("No ET data loaded. Please fetch data first.")
        return

    df = st.session_state['et_data']
    variable = params['variable']
    interval = params['interval']
    start_date = params['start_date']
    end_date = params['end_date']
    units = params['units']

    st.header("Results")
    st.subheader("ET Time-Series")

    fig = px.line(
        df,
        x='date',
        y=variable,
        title=f"{params['model']} {variable} ({units})",
        labels={'date': 'Date', variable: f"{variable} ({units})"}
    )
    fig.update_layout(
        hovermode="x unified",
        plot_bgcolor="rgba(240,249,255,0.6)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Basic summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", f"{df[variable].sum():.1f} {units}")
    with col2:
        st.metric("Average", f"{df[variable].mean():.2f} {units}/{interval}")
    with col3:
        st.metric("Maximum", f"{df[variable].max():.2f} {units}")
    with col4:
        st.metric("Count", f"{len(df)}")

    # Additional placeholders for advanced irrigation logic, etc.
    st.subheader("Irrigation Recommendations (Optional)")
    # ...
    # For demonstration, you could call get_irrigation_recommendation() here
    # or simply show placeholders.

def main():
    """Main entry point for the Streamlit app."""
    initialize_session_state()
    st.title(APP_TITLE)
    st.markdown("Satellite-based irrigation recommendations for optimizing water use in agriculture")

    # Render sidebar & store user parameters
    params = render_sidebar()

    # 1) Field selection
    render_map_section()

    # 2) Single bottom-located button to fetch data
    st.write("---")
    st.subheader("Fetch OpenET Data")
    fetch_disabled = (st.session_state['user_polygon'] is None or not st.session_state['api_key'])
    if st.button("Fetch ET Data", disabled=fetch_disabled, type="primary"):
        if params['end_date'] < params['start_date']:
            st.error("End date cannot be before start date.")
        else:
            fetch_data(params)

    # 3) Show results if data is available
    render_results(params)

    st.write("---")
    st.caption("Powered by [OpenET](https://etdata.org) | ESRI World Imagery basemap | Geoapify address search")

if __name__ == "__main__":
    main()
