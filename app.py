import os
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw, MeasureControl
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import List
import logging
from dotenv import load_dotenv
import requests

# Load local modules
from openet_api import fetch_openet_data, OpenETError, load_api_key
from data_cache import load_cache, save_cache
from irrigation import get_irrigation_recommendation
from utils import validate_geometry, calculate_area, format_coordinates_for_display

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App configuration
APP_TITLE = "OpenET Irrigation Advisory"
# Default center: 402 West State Farm Road, North Platte, NE
DEFAULT_CENTER = [41.121092, -100.768147]
DEFAULT_ZOOM = 6
MAX_AREA_ACRES = 50000

# Attempt to load API keys from environment
OPENET_API_KEY = os.environ.get('OPENET_API_KEY', '')
# IMPORTANT: for address search
GEOAPIFY_API_KEY = os.environ.get('GEOAPIFY_API_KEY', '')

# Set page configuration with custom theme colors
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Cleaned up) ---
st.markdown("""
<style>
    /* Space around main container */
    .main .block-container {
        padding-top: 2rem;
    }

    /* Headings */
    h1 { color: #1e5b94; }
    h2 {
        color: #1e5b94;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    h3 { color: #1e5b94; font-size: 1.3rem; }

    /* Button styling */
    .stButton>button {
        background-color: #1e5b94;
        color: white;
    }
    .stButton>button:hover {
        background-color: #164576;
    }

    /* Additional Streamlit color classes */
    .st-bb { border-bottom-color: #1e5b94; }
    .st-at { background-color: #1e5b94; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #173d5e;
    }
    [data-testid="stSidebar"] .st-bq {
        color: white !important;
    }
    [data-testid="stSidebar"] .st-c0 {
        color: #b8d1e6 !important;
    }
    [data-testid="stSidebar"] h1 {
        color: white !important;
    }
    [data-testid="stSidebar"] h2 {
        color: white !important;
        border-color: #315b7c !important;
    }
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSubheader {
        color: #8aadce !important;
        font-weight: 600 !important;
        margin-top: 20px !important;
    }
    [data-testid="stSidebar"] p {
        color: #e0eaf2 !important;
    }
    [data-testid="stSidebar"] label {
        color: #e0eaf2 !important;
    }

    /* Field selection box */
    .selected-field-box {
        background-color: #1e5b94;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .selected-field-box h3 {
        color: white !important;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .selected-field-box p {
        color: white;
        margin-bottom: 0.5rem;
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
    .instructions-box h3 {
        color: white !important;
        margin-top: 0;
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
    .info-box {
        background-color: #f0f6fb;
        border-left: 5px solid #4287f5;
    }
    .success-box {
        background-color: #f0f9f5;
        border-left: 5px solid #13ab5c;
    }
    .warning-box {
        background-color: #fffaf0;
        border-left: 5px solid #f7b034;
    }
    .error-box {
        background-color: #fef2f1;
        border-left: 5px solid #ef564d;
    }

    /* Space optimization */
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0;
    }

    /* General button styling for the app */
    div.stButton > button:first-child {
        border-radius: 4px;
        font-weight: 500;
        border: none;
        padding: 0.4rem 1rem;
    }
    div.stButton > button:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)
# --- END CUSTOM CSS ---

@st.cache_data(ttl=3600*24)  # Cache for 24 hours
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

def render_sidebar():
    """Render the sidebar with app information and data controls."""
    st.sidebar.title("OpenET Irrigation Advisory")
    
    with st.sidebar.expander("API Key Settings", expanded=False):
        api_key = st.text_input(
            "OpenET API Key", 
            value=st.session_state.get('api_key', ''),
            type="password",
            help="Required to fetch data from OpenET. Sign up at https://etdata.org"
        )
        st.session_state['api_key'] = api_key
        if not api_key:
            st.warning("API key required to fetch data")
            
    st.sidebar.subheader("Time Period")
    
    def update_start_date():
        st.session_state['start_date'] = st.session_state['start_date_input'].isoformat()
    def update_end_date():
        st.session_state['end_date'] = st.session_state['end_date_input'].isoformat()
    
    today = datetime.now().date()
    future_date = today + timedelta(days=365*10)
    
    default_start = datetime.fromisoformat(st.session_state['start_date']).date()
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=default_start,
        min_value=datetime(1985, 1, 1).date(), 
        max_value=future_date,
        key="start_date_input",
        on_change=update_start_date
    )
    
    default_end = datetime.fromisoformat(st.session_state['end_date']).date()
    end_date = st.sidebar.date_input(
        "End Date", 
        value=default_end,
        min_value=datetime(1985, 1, 1).date(),
        max_value=future_date,
        key="end_date_input",
        on_change=update_end_date
    )
    
    if end_date < start_date:
        st.sidebar.error("End date cannot be before start date")
        end_date = start_date
    if end_date > today:
        st.sidebar.warning("The OpenET API does not return data beyond today's date.")
    
    with st.sidebar.expander("ET Settings", expanded=False):
        st.subheader("Data Parameters")
        interval = st.selectbox("Data Interval", ["daily", "monthly"], index=0)
        model = st.selectbox(
            "ET Model", 
            options=["Ensemble", "SSEBop", "SIMS", "PTJPL", "eeMETRIC", "DisALEXI"],
            index=0
        )
        variable = st.selectbox(
            "Variable", 
            options=["ET", "ETo", "ETof", "NDVI", "PR"],
            index=0
        )
        units = st.selectbox(
            "Units", 
            options=["mm", "in"],
            index=0
        )
    
    if st.session_state['et_data'] is not None:
        st.sidebar.subheader("Irrigation Settings")
        mode = st.sidebar.selectbox("Irrigation Mode", ["daily", "threshold"], index=0)
        if mode == "threshold":
            threshold = st.sidebar.number_input("Depletion Threshold (mm)", min_value=10, max_value=100, value=30)
        else:
            threshold = 25
        efficiency = st.sidebar.slider("Irrigation Efficiency (%)", 50, 100, 85)
        display_units = st.sidebar.selectbox(
            "Display Units", 
            options=["mm", "inches"],
            index=0 if units == "mm" else 1
        )
    else:
        mode = "daily"
        threshold = 25
        efficiency = 85
        display_units = units
    
    with st.sidebar.expander("About", expanded=False):
        st.markdown("""
        This app uses [OpenET](https://etdata.org) data to provide irrigation 
        recommendations for agricultural fields based on satellite-derived 
        evapotranspiration (ET) measurements.
        
        **Disclaimer:** Recommendations are estimates based on ET data
        and should be used as a guide only. Actual needs may vary
        by field conditions and other factors.
        """)
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'interval': interval,
        'model': model,
        'variable': variable,
        'units': units,
        'mode': mode,
        'threshold': threshold,
        'efficiency': efficiency / 100.0,
        'display_units': display_units
    }

def render_map_section():
    """
    Simplified Geoapify address search:
    Automatically uses the first address match to re-center the map.
    """
    st.header("Field Selection")

    address_col, button_col = st.columns([4, 1])
    with address_col:
        address_query = st.text_input(
            "Address Input",
            "",
            placeholder="402 West State Farm Road, North Platte, NE",
            label_visibility="collapsed"
        )
    with button_col:
        search_clicked = st.button("Search", key="address_search_button")

    if search_clicked and address_query and GEOAPIFY_API_KEY:
        geo_url = f"https://api.geoapify.com/v1/geocode/autocomplete?text={address_query}&apiKey={GEOAPIFY_API_KEY}"
        try:
            resp = requests.get(geo_url, timeout=10)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            if not features:
                st.warning("No address matches found.")
            else:
                # Automatically use the first address candidate
                first_candidate = features[0]
                lat = first_candidate["properties"]["lat"]
                lon = first_candidate["properties"]["lon"]
                st.session_state['map_center'] = [lat, lon]
                st.session_state['map_zoom'] = 18
                st.experimental_rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"Address lookup failed: {e}")
    elif search_clicked and not GEOAPIFY_API_KEY:
        st.warning("No GEOAPIFY_API_KEY found in environment. Cannot perform address search.")

    map_col, control_col = st.columns([4, 1])
    with map_col:
        m = create_map()
        map_data = st_folium(m, height=450, width="100%")
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
        st.markdown("""
        <div class="instructions-box" style="font-size:0.9rem;">
            <b>How to Select a Field:</b>
            <ol style="padding-left:1rem;">
                <li>Use rectangle ⬜ or polygon 🔺 tools.</li>
                <li>Draw your boundary.</li>
                <li>Field highlights after drawing.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state['user_polygon']:
            if st.button("Clear Selection", key="clear_selection"):
                st.session_state['user_polygon'] = None
                st.session_state['geometry_metadata'] = None
                st.session_state['et_data'] = None
                st.session_state['last_query_params'] = None
                st.experimental_rerun()
            if st.button("Fetch ET Data", key="fetch_data"):
                fetch_data(initialize_session_state())

def fetch_data(params):
    """Fetch ET data based on parameters."""
    if st.session_state['user_polygon'] is None:
        st.warning("Please select a field on the map first.")
        return
    if not st.session_state['api_key']:
        st.warning("An API key is required to fetch data from OpenET. Please enter your API key in the sidebar.")
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
    if st.session_state['last_query_params'] == current_params and st.session_state['et_data'] is not None:
        st.success("Using previously fetched data.")
        return
    with st.spinner("Retrieving data from satellite measurements..."):
        try:
            today = datetime.now().date()
            df = get_et_data_cached(
                geometry=st.session_state['user_polygon'],
                start_date=start_date.isoformat(),
                end_date=min(end_date, today).isoformat(),
                interval=params['interval'],
                model=params['model'],
                variable=params['variable'],
                units=params['units']
            )
            if df is None or df.empty:
                st.error("No data returned. This may be due to being outside the coverage area or no data for that period.")
            else:
                st.session_state['et_data'] = df
                st.session_state['last_query_params'] = current_params
                st.success(f"Data retrieved: {len(df)} records from {start_date} to {min(end_date, today)}")
        except OpenETError as e:
            st.error(f"Error fetching data: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

def render_results(params):
    """Render results and irrigation recommendations."""
    if st.session_state['et_data'] is None:
        return
    df = st.session_state['et_data']
    variable = params['variable']
    interval = params['interval']
    start_date = params['start_date']
    end_date = params['end_date']
    units = params['units']
    display_units = params['display_units']
    mode = params['mode']
    threshold = params['threshold']
    efficiency = params['efficiency']
    
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
        font=dict(color="#333333"),
        margin=dict(t=50, b=50),
        xaxis=dict(
            gridcolor="rgba(220,230,240,0.5)",
            title_font=dict(size=14)
        ),
        yaxis=dict(
            gridcolor="rgba(220,230,240,0.5)",
            title_font=dict(size=14)
        )
    )
    if interval == "daily" and len(df) > 31:
        st.plotly_chart(fig, use_container_width=True)
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly['date'].dt.to_period('M')
        monthly_values = df_monthly.groupby('month')[variable].sum().reset_index()
        monthly_values['month'] = monthly_values['month'].dt.to_timestamp()
        fig2 = px.bar(
            monthly_values, 
            x='month', 
            y=variable,
            title=f"Monthly Total {variable} ({units})"
        )
        fig2.update_layout(
            hovermode="x unified",
            plot_bgcolor="rgba(240,249,255,0.6)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#333333"),
            margin=dict(t=50, b=50),
            xaxis=dict(
                gridcolor="rgba(220,230,240,0.5)",
                title_font=dict(size=14)
            ),
            yaxis=dict(
                gridcolor="rgba(220,230,240,0.5)",
                title_font=dict(size=14)
            )
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", f"{df[variable].sum():.1f} {units}")
    with col2:
        st.metric("Average", f"{df[variable].mean():.2f} {units}/{interval}")
    with col3:
        st.metric("Maximum", f"{df[variable].max():.2f} {units}")
    with col4:
        st.metric("Count", f"{len(df)}")
    
    st.subheader("Irrigation Recommendation")
    rain_col = None
    if "PR" in df.columns:
        rain_col = "PR"
    elif "Rain" in df.columns:
        rain_col = "Rain"
    
    try:
        recommendation = get_irrigation_recommendation(
            df=df,
            mode=mode,
            threshold_mm=threshold,
            system_efficiency=efficiency,
            et_col=variable,
            rain_col=rain_col,
            units=display_units
        )
        if recommendation["status"] == "success":
            st.markdown(f"""
            <div class="status-box info-box">
                <h4>Recommendation</h4>
                {recommendation["recommendation"]}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="status-box success-box">
                <h4>Summary</h4>
                <p>Period analyzed: {recommendation["period_days"]} days<br>
                Total crop water use (ET): {recommendation["total_et"]:.1f} {display_units}<br>
                {f"Total rainfall: {recommendation['total_rain']:.1f} {display_units}<br>" if recommendation["total_rain"] > 0 else ""}
                Total irrigation need: {recommendation["total_irrigation"]:.1f} {display_units}</p>
            </div>
            """, unsafe_allow_html=True)
            if mode == "threshold" and "schedule" in recommendation and recommendation["schedule"]:
                sched_df = pd.DataFrame(recommendation["schedule"])
                st.markdown("#### Irrigation Schedule")
                fig_sched = go.Figure(data=[
                    go.Table(
                        header=dict(
                            values=["Date", f"Amount ({display_units})"],
                            fill_color='#1e5b94',
                            font=dict(color='white'),
                            align='left'
                        ),
                        cells=dict(
                            values=[sched_df['date'], sched_df['amount']],
                            fill_color=['#f5f8fc', '#f5f8fc'],
                            align='left'
                        )
                    )
                ])
                fig_sched.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                st.plotly_chart(fig_sched, use_container_width=True)
                if interval == "daily":
                    df_cum = df.copy()
                    df_cum['Cumulative_ET'] = df_cum[variable].cumsum()
                    if rain_col:
                        df_cum['Net_ET'] = (df_cum[variable] - df_cum[rain_col]).clip(lower=0)
                        df_cum['Cumulative_Net_ET'] = df_cum['Net_ET'].cumsum()
                        df_cum['Cumulative_Rain'] = df_cum[rain_col].cumsum()
                    if display_units != units:
                        conv = 1/25.4 if display_units == "inches" else 25.4
                        for col in ['Cumulative_ET','Net_ET','Cumulative_Net_ET','Cumulative_Rain']:
                            if col in df_cum.columns:
                                df_cum[col] = df_cum[col] * conv
                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=df_cum['date'],
                        y=df_cum['Cumulative_ET'],
                        mode='lines', name=f"Cumulative {variable}",
                        line=dict(color='#1e5b94', width=2)
                    ))
                    if rain_col:
                        fig_cum.add_trace(go.Scatter(
                            x=df_cum['date'],
                            y=df_cum['Cumulative_Rain'],
                            mode='lines', name="Cumulative Rain",
                            line=dict(color='#13ab5c', width=2)
                        ))
                        fig_cum.add_trace(go.Scatter(
                            x=df_cum['date'],
                            y=df_cum['Cumulative_Net_ET'],
                            mode='lines', name='Cumulative Net ET',
                            line=dict(color='#f7b034', width=2)
                        ))
                    fig_cum.update_layout(
                        title=f"Cumulative Water Balance ({display_units})",
                        xaxis_title="Date",
                        yaxis_title=f"Cumulative Amount ({display_units})",
                        hovermode="x unified",
                        plot_bgcolor="rgba(240,249,255,0.6)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)
        else:
            st.error(recommendation["message"])
    except Exception as e:
        st.error(f"Error generating irrigation recommendation: {str(e)}")

def main():
    """Main Streamlit app entry point."""
    initialize_session_state()
    st.title(APP_TITLE)
    st.markdown("Satellite-based irrigation recommendations for optimizing water use in agriculture")
    params = render_sidebar()
    render_map_section()
    if st.session_state['user_polygon'] is not None:
        fetch_button_col1, fetch_button_col2 = st.columns([1, 3])
        with fetch_button_col1:
            if st.button("Fetch ET Data", key="fetch_button", type="primary"):
                fetch_data(params)
        if st.session_state['et_data'] is not None:
            render_results(params)
    st.markdown("---")
    st.caption("Powered by [OpenET](https://openetdata.org) | ESRI World Imagery basemap | Geoapify address search")

if __name__ == "__main__":
    main()
