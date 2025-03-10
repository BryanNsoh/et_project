"""
OpenET Irrigation Advisory - Streamlit App
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

# Load local modules
from openet_api import fetch_openet_data, OpenETError, load_api_key
from data_cache import load_cache, save_cache, clear_cache, get_cache_info
from irrigation import get_irrigation_recommendation, summarize_irrigation_needs
from utils import (
    process_uploaded_geometry, 
    validate_geometry, 
    calculate_area, 
    format_coordinates_for_display,
    get_random_demo_field
)

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App configuration
APP_TITLE = "OpenET Irrigation Advisory Demo"
DEFAULT_CENTER = [37.0, -120.0]  # California Central Valley
DEFAULT_ZOOM = 6
MAX_AREA_ACRES = 50000

# Attempt to load API keys from environment (including Geoapify if present)
OPENET_API_KEY = os.environ.get('OPENET_API_KEY', '')
GEOAPIFY_API_KEY = os.environ.get('GEOAPIFY_API_KEY', '')

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit Cache for API calls
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
    
    if 'show_demo' not in st.session_state:
        st.session_state['show_demo'] = False
    
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = OPENET_API_KEY  # pre-load if found
    
    # Default date values
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = last_year.isoformat()
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = today.isoformat()
    
    # Default map center & zoom (can be updated by address search)
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
        if not isinstance(coords[0], (int, float)):  # It's a polygon
            folium.Polygon(
                locations=coords,
                color='blue',
                fill=True,
                fill_color='blue',
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
    """Render the map section for field selection + optional address search."""
    st.subheader("1. Select Your Field")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create and display the map
        m = create_map()
        map_data = st_folium(m, height=500, width=700)
        
        # Process user drawings
        if map_data and map_data.get("all_drawings") is not None:
            drawings = map_data["all_drawings"]
            if drawings:
                if len(drawings) > 1:
                    st.info("Multiple shapes drawn. Using the most recent one.")
                
                last_drawing = drawings[-1]
                geometry = last_drawing.get("geometry", {})
                geom_type = geometry.get("type", "")
                
                if geom_type == "Point":
                    coords = geometry.get("coordinates", [])
                    st.session_state['user_polygon'] = coords
                    st.session_state['geometry_metadata'] = {
                        "source": "draw",
                        "geometry_type": "point"
                    }
                elif geom_type in ["Polygon", "Rectangle"]:
                    coords = geometry.get("coordinates", [[]])[0]
                    
                    try:
                        validate_geometry(coords, "polygon")
                        area_sq_m = calculate_area(coords)
                        area_acres = area_sq_m * 0.000247105
                        if area_acres > MAX_AREA_ACRES:
                            st.error(f"Area exceeds maximum allowed size of {MAX_AREA_ACRES} acres. Please select a smaller area.")
                        else:
                            st.session_state['user_polygon'] = coords
                            st.session_state['geometry_metadata'] = {
                                "source": "draw",
                                "geometry_type": "polygon",
                                "area_acres": area_acres,
                                "area_sq_m": area_sq_m
                            }
                    except ValueError as e:
                        st.error(str(e))
    
    with col2:
        st.write("**Draw your field** using the polygon or rectangle tool, or **upload** a boundary file.\n")
        
        # Address search (optional)
        st.markdown("**Optional: Search by Address**")
        address_query = st.text_input("Enter address (autocomplete via Geoapify):", "")
        
        if address_query and GEOAPIFY_API_KEY:
            # Query Geoapify autocomplete
            geo_url = f"https://api.geoapify.com/v1/geocode/autocomplete?text={address_query}&apiKey={GEOAPIFY_API_KEY}"
            try:
                resp = requests.get(geo_url, timeout=10)
                resp.raise_for_status()
                features = resp.json().get("features", [])
                
                if not features:
                    st.warning("No address matches found.")
                else:
                    # Build suggestion labels
                    suggestions = [feat["properties"]["formatted"] for feat in features]
                    selected_addr = st.selectbox("Select the correct address:", options=suggestions)
                    
                    if st.button("Locate"):
                        # Find that feature
                        chosen = None
                        for feat in features:
                            if feat["properties"]["formatted"] == selected_addr:
                                chosen = feat
                                break
                        if chosen:
                            lat = chosen["properties"]["lat"]
                            lon = chosen["properties"]["lon"]
                            # Update map center & zoom
                            st.session_state['map_center'] = [lat, lon]
                            st.session_state['map_zoom'] = 18
                            st.experimental_rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Address lookup failed: {e}")
        elif address_query and not GEOAPIFY_API_KEY:
            st.warning("No GEOAPIFY_API_KEY found in environment. Cannot perform address search.")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload boundary file", 
            type=["zip", "geojson", "kml"]
        )
        if uploaded_file:
            try:
                coords, metadata = process_uploaded_geometry(uploaded_file, MAX_AREA_ACRES)
                st.session_state['user_polygon'] = coords
                st.session_state['geometry_metadata'] = metadata
                st.success(f"Boundary loaded successfully: {format_coordinates_for_display(coords)}")
                
                # Show area for polygons
                if metadata.get("geometry_type") == "polygon" and "area_acres" in metadata:
                    st.info(f"Field area: {metadata['area_acres']:.2f} acres ({metadata['area_sq_m']:.2f} sq meters)")
                
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        # Demo field button
        if st.button("Use Demo Field"):
            demo_field = get_random_demo_field()
            area_sq_m = calculate_area(demo_field)
            area_acres = area_sq_m * 0.000247105
            
            st.session_state['user_polygon'] = demo_field
            st.session_state['geometry_metadata'] = {
                "source": "demo",
                "geometry_type": "polygon",
                "area_acres": area_acres,
                "area_sq_m": area_sq_m
            }
            st.success("Demo field loaded!")
            st.experimental_rerun()
    
    # Display info about selected field
    if st.session_state['user_polygon'] is not None and st.session_state['geometry_metadata'] is not None:
        coords = st.session_state['user_polygon']
        metadata = st.session_state['geometry_metadata']
        
        st.write("---")
        st.write("**Selected field information:**")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write(format_coordinates_for_display(coords))
            if metadata.get("geometry_type") == "polygon" and "area_acres" in metadata:
                st.write(f"Area: {metadata['area_acres']:.2f} acres ({metadata['area_sq_m']:.2f} sq meters)")
            if metadata.get("source") == "upload":
                st.write(f"Source: Uploaded file ({metadata.get('filename', 'unknown')})")
            elif metadata.get("source") == "draw":
                st.write("Source: Drawn on map")
            elif metadata.get("source") == "demo":
                st.write("Source: Demo field")
        
        with c2:
            if st.button("Clear Selection"):
                st.session_state['user_polygon'] = None
                st.session_state['geometry_metadata'] = None
                st.session_state['et_data'] = None
                st.session_state['last_query_params'] = None
                st.experimental_rerun()

def render_data_parameters():
    """Render the data parameters section for OpenET retrieval."""
    st.subheader("2. Set Data Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    # Use session state for date inputs
    def update_start_date():
        st.session_state['start_date'] = st.session_state['start_date_input'].isoformat()
    def update_end_date():
        st.session_state['end_date'] = st.session_state['end_date_input'].isoformat()
    
    today = datetime.now().date()
    future_date = today + timedelta(days=365*10)  # allow selection up to 10 years future
    
    with col1:
        default_start = datetime.fromisoformat(st.session_state['start_date']).date()
        start_date = st.date_input(
            "Start Date", 
            value=default_start,
            min_value=datetime(1985, 1, 1).date(), 
            max_value=future_date,
            key="start_date_input",
            on_change=update_start_date
        )
    
    with col2:
        default_end = datetime.fromisoformat(st.session_state['end_date']).date()
        end_date = st.date_input(
            "End Date", 
            value=default_end,
            min_value=datetime(1985, 1, 1).date(),
            max_value=future_date,
            key="end_date_input",
            on_change=update_end_date
        )
        if end_date < start_date:
            st.error("End date cannot be before start date")
            end_date = start_date
        if end_date > today:
            st.warning("FUTURE DATES NOT SUPPORTED: The OpenET API does not return data beyond today's date.")
    
    with col3:
        interval = st.selectbox(
            "Data Interval", 
            options=["daily", "monthly"],
            index=0
        )
    
    c1, c2, c3 = st.columns(3)
    with c1:
        model = st.selectbox(
            "ET Model", 
            options=["Ensemble", "SSEBop", "SIMS", "PTJPL", "eeMETRIC", "DisALEXI"],
            index=0
        )
    with c2:
        variable = st.selectbox(
            "Variable", 
            options=["ET", "ETo", "ETof", "NDVI", "PR"],
            index=0
        )
    with c3:
        units = st.selectbox(
            "Units", 
            options=["mm", "in"],
            index=0
        )
    
    # API key input
    with st.expander("API Key Settings"):
        api_key = st.text_input(
            "OpenET API Key", 
            value=st.session_state.get('api_key', ''),
            type="password"
        )
        st.session_state['api_key'] = api_key
        if not api_key:
            st.warning("An API key is required to fetch data from OpenET. Sign up at https://etdata.org")
    
    # Prepare parameters for data fetch
    current_params = (
        str(st.session_state['user_polygon']), 
        start_date.isoformat(),
        end_date.isoformat(),
        interval,
        model,
        variable,
        units
    )
    
    # Fetch button
    fetch_disabled = (
        st.session_state['user_polygon'] is None or 
        not st.session_state['api_key']
    )
    
    if st.button("Fetch ET Data", disabled=fetch_disabled, type="primary"):
        if end_date > today:
            st.error("Cannot fetch data for a future date. Please select an end date no later than today.")
        elif st.session_state['last_query_params'] == current_params and st.session_state['et_data'] is not None:
            st.success("Using previously fetched data. (Parameters unchanged.)")
        else:
            with st.spinner("Fetching data from OpenET..."):
                try:
                    df = get_et_data_cached(
                        geometry=st.session_state['user_polygon'],
                        start_date=start_date.isoformat(),
                        end_date=min(end_date, today).isoformat(),
                        interval=interval,
                        model=model,
                        variable=variable,
                        units=units
                    )
                    if df is None or df.empty:
                        st.error("No data returned. Possibly out of coverage or no data for that period.")
                    else:
                        st.session_state['et_data'] = df
                        st.session_state['last_query_params'] = current_params
                        st.success(f"Data retrieved: {len(df)} records from {start_date} to {min(end_date, today)}")
                except OpenETError as e:
                    st.error(f"Error fetching data: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
    
    return start_date, end_date, interval, model, variable, units

def render_results(start_date, end_date, interval, model, variable, units):
    """Render results and irrigation recommendations."""
    if st.session_state['et_data'] is None:
        return
    
    st.subheader("3. Results and Irrigation Recommendations")
    df = st.session_state['et_data']
    
    tab1, tab2, tab3 = st.tabs(["ET Visualization", "Irrigation Recommendation", "Data Table"])
    
    with tab1:
        st.write(f"**{variable} Time-Series** ({interval}, {start_date} to {end_date})")
        fig = px.line(
            df, 
            x='date', 
            y=variable,
            title=f"{model} {variable} ({units})",
            labels={'date': 'Date', variable: f"{variable} ({units})"}
        )
        fig.update_layout(hovermode="x unified")
        
        if interval == "daily" and len(df) > 31:
            # Show monthly aggregation side-by-side
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
            fig2.update_layout(hovermode="x unified")
            
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)
        
        # Basic summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", f"{df[variable].sum():.1f} {units}")
        with col2:
            st.metric("Average", f"{df[variable].mean():.2f} {units}/{interval}")
        with col3:
            st.metric("Maximum", f"{df[variable].max():.2f} {units}")
        with col4:
            st.metric("Count", f"{len(df)}")
    
    with tab2:
        st.write("**Irrigation Recommendation**")
        mode = st.selectbox("Irrigation Mode", ["daily", "threshold"], index=0)
        
        if mode == "threshold":
            threshold = st.number_input("Depletion Threshold (mm)", min_value=10, max_value=100, value=30)
        else:
            threshold = 25  # default, not used for daily
        
        colA, colB = st.columns(2)
        with colA:
            efficiency = st.slider("Irrigation Efficiency (%)", 50, 100, 85) / 100.0
        with colB:
            display_units = st.selectbox(
                "Display Units", 
                options=["mm", "inches"],
                index=0 if units == "mm" else 1
            )
        
        rain_col = "PR" if "PR" in df.columns else "Rain" if "Rain" in df.columns else None
        
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
                st.info(recommendation["recommendation"])
                st.success(summarize_irrigation_needs(recommendation))
                
                # If threshold mode, show schedule
                if mode == "threshold" and "schedule" in recommendation:
                    schedule_list = recommendation["schedule"]
                    if schedule_list:
                        sched_df = pd.DataFrame(schedule_list)
                        fig_sched = go.Figure(data=[
                            go.Table(
                                header=dict(values=["Date", f"Amount ({display_units})"], fill_color='paleturquoise', align='left'),
                                cells=dict(values=[sched_df['date'], sched_df['amount']], fill_color='lavender', align='left')
                            )
                        ])
                        st.plotly_chart(fig_sched, use_container_width=True)
                    else:
                        st.write("No irrigation events scheduled for this period.")
                
                # Cumulative chart if daily
                if interval == "daily":
                    df_cum = df.copy()
                    df_cum['Cumulative_ET'] = df_cum[variable].cumsum()
                    if rain_col:
                        df_cum['Net_ET'] = (df_cum[variable] - df_cum[rain_col]).clip(lower=0)
                        df_cum['Cumulative_Net_ET'] = df_cum['Net_ET'].cumsum()
                        df_cum['Cumulative_Rain'] = df_cum[rain_col].cumsum()
                    
                    # Unit conversion if needed
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
                        line=dict(color='red', width=2)
                    ))
                    if rain_col:
                        fig_cum.add_trace(go.Scatter(
                            x=df_cum['date'],
                            y=df_cum['Cumulative_Rain'],
                            mode='lines', name=f"Cumulative Rain",
                            line=dict(color='blue', width=2)
                        ))
                        fig_cum.add_trace(go.Scatter(
                            x=df_cum['date'],
                            y=df_cum['Cumulative_Net_ET'],
                            mode='lines', name='Cumulative Net ET',
                            line=dict(color='orange', width=2)
                        ))
                    fig_cum.update_layout(
                        title=f"Cumulative Water Balance ({display_units})",
                        xaxis_title="Date",
                        yaxis_title=f"Cumulative Amount ({display_units})",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)
            
            else:
                st.error(recommendation["message"])
        
        except Exception as e:
            st.error(f"Error generating irrigation recommendation: {str(e)}")
    
    with tab3:
        st.write("**Raw Data Table**")
        display_df = df.copy()
        if units != display_units:
            conv = 1/25.4 if display_units == "inches" else 25.4
            numeric_cols = [col for col in display_df.columns if col != "date" and pd.api.types.is_numeric_dtype(display_df[col])]
            for col in numeric_cols:
                display_df[col] = display_df[col] * conv
        
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_df, use_container_width=True)
        
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv_data,
            f"openet_{variable}_{start_date}_to_{end_date}.csv",
            "text/csv",
            key='download-csv'
        )

def render_sidebar():
    """Render the sidebar with app information and settings."""
    st.sidebar.title("OpenET Irrigation Advisory")
    st.sidebar.markdown("""
    This app demonstrates how to use [OpenET](https://etdata.org) data
    to provide irrigation recommendations for agricultural fields.
    """)
    st.sidebar.subheader("How to Use")
    st.sidebar.markdown("""
    1. Draw or upload your field boundary.
    2. (Optional) Use Geoapify address search to jump to a location.
    3. Set date range & data parameters.
    4. Click "Fetch ET Data".
    5. View results & irrigation recommendations.
    """)
    st.sidebar.subheader("About")
    st.sidebar.markdown("""
    OpenET provides satellite-based evapotranspiration (ET) data for
    estimating crop water use. This demo illustrates basic irrigation
    scheduling logic.
    """)
    st.sidebar.subheader("Disclaimer")
    st.sidebar.info("""
    The irrigation recommendations are estimates based on ET data
    and should be used as a guide only. Actual needs may vary
    by field conditions and other factors.
    """)
    st.sidebar.markdown("[Learn more about OpenET](https://etdata.org)")
    st.sidebar.caption("App Version: 1.0.1")

def main():
    """Main Streamlit app entry point."""
    initialize_session_state()
    st.title(APP_TITLE)
    render_sidebar()
    
    render_map_section()
    if st.session_state['user_polygon'] is not None:
        start_date, end_date, interval, model, variable, units = render_data_parameters()
        if st.session_state['et_data'] is not None:
            render_results(start_date, end_date, interval, model, variable, units)
    
    with st.expander("Cache Information", expanded=False):
        st.write("Local cache info (for debugging):")
        info = get_cache_info()
        st.write(f"Entries: {info.get('count',0)}")
        st.write(f"Total size: {info.get('total_size_kb',0.0):.2f} KB")
        if st.button("Clear Cache"):
            if clear_cache():
                st.success("Cache cleared.")
                st.cache_data.clear()
            else:
                st.error("Failed to clear cache.")
    
    st.write("---")
    st.caption("Powered by OpenET API | ESRI World Imagery basemap | Geoapify address search")

if __name__ == "__main__":
    main()
