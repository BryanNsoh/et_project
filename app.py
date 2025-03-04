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
    # Try to load from local cache first
    df = load_cache(
        coords=geometry,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        model=model,
        variable=variable,
        max_age_hours=24  # Consider cache expired after 24 hours
    )
    
    if df is not None:
        return df
    
    # If not in cache, call API
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
        
        # Save to cache
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
        # Re-raise as a non-cached exception
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
        # Try to get from environment
        st.session_state['api_key'] = os.environ.get('OPENET_API_KEY', '')
        
    # Initialize date values
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = last_year.isoformat()
    
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = today.isoformat()

def create_map(center=DEFAULT_CENTER, zoom=DEFAULT_ZOOM):
    """
    Create an interactive map with drawing tools.
    
    Returns:
        folium.Map: The map object
    """
    m = folium.Map(location=center, zoom_start=zoom, control_scale=True)
    
    # Add tile layers
    folium.TileLayer('openstreetmap').add_to(m)
    folium.TileLayer('Stamen Terrain').add_to(m)
    folium.TileLayer('CartoDB positron').add_to(m)
    
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
    
    # Add legend or info
    if st.session_state['user_polygon'] is not None:
        # If a polygon is already selected, show it on the map
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
            sw = min(lat for _, lat in coords), min(lon for lon, _ in coords)
            ne = max(lat for _, lat in coords), max(lon for lon, _ in coords)
            m.fit_bounds([sw, ne])
    
    return m

def render_map_section():
    """Render the map section for field selection."""
    st.subheader("1. Select Your Field")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create and display the map
        m = create_map()
        map_data = st_folium(m, height=500, width=700)
        
        # Process drawing results if available
        if map_data and map_data.get("all_drawings") is not None:
            drawings = map_data["all_drawings"]
            if drawings and len(drawings) > 0:
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
                        # Validate the geometry
                        validate_geometry(coords, "polygon")
                        
                        # Calculate area
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
        st.write("**Draw your field on the map** or upload a boundary file.")
        st.write("Use the polygon or rectangle tool to draw. Double-click to finish drawing.")
        
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
                
                # Trigger a rerun to show the polygon on the map
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        # Option to use demo field
        if st.button("Use Demo Field"):
            demo_field = get_random_demo_field()
            
            # Calculate area
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
            st.rerun()
    
    # Display information about selected field
    if st.session_state['user_polygon'] is not None and st.session_state['geometry_metadata'] is not None:
        coords = st.session_state['user_polygon']
        metadata = st.session_state['geometry_metadata']
        
        st.write("---")
        st.write("**Selected field information:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(format_coordinates_for_display(coords))
            if metadata.get("geometry_type") == "polygon" and "area_acres" in metadata:
                st.write(f"Area: {metadata['area_acres']:.2f} acres ({metadata['area_sq_m']:.2f} sq meters)")
            
            if metadata.get("source") == "upload":
                st.write(f"Source: Uploaded file ({metadata.get('filename', 'unknown')})")
            elif metadata.get("source") == "draw":
                st.write("Source: Drawn on map")
            elif metadata.get("source") == "demo":
                st.write("Source: Demo field")
        
        with col2:
            # Button to clear selection
            if st.button("Clear Selection"):
                st.session_state['user_polygon'] = None
                st.session_state['geometry_metadata'] = None
                st.session_state['et_data'] = None
                st.session_state['last_query_params'] = None
                st.rerun()

def render_data_parameters():
    """Render the data parameters section."""
    st.subheader("2. Set Data Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    # Save dates in session state to handle dependencies properly
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = (datetime.now().date() - timedelta(days=365)).isoformat()
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = datetime.now().date().isoformat()
        
    # Callback functions to update session state
    def update_start_date():
        start = st.session_state['start_date_input']
        st.session_state['start_date'] = start.isoformat()
        
    def update_end_date():
        end = st.session_state['end_date_input']
        st.session_state['end_date'] = end.isoformat()
        
    with col1:
        # Date range selection
        today = datetime.now().date()
        last_year = today - timedelta(days=365)
        
        # Convert stored string dates back to date objects
        default_start = datetime.fromisoformat(st.session_state['start_date']).date()
        
        # Future date for consistency with end date
        future_date = today + timedelta(days=365*10)  # 10 years in the future
        
        start_date = st.date_input(
            "Start Date", 
            value=default_start,
            min_value=datetime(1985, 1, 1).date(),  # OpenET data starts around 1985
            max_value=future_date,  # Allow future dates for forecasting scenarios
            help="Select the start date for ET data",
            key="start_date_input",
            on_change=update_start_date
        )
    
    with col2:
        # Convert stored string dates back to date objects
        default_end = datetime.fromisoformat(st.session_state['end_date']).date()
        
        # Allow end dates into the future (10 years from now)
        future_date = today + timedelta(days=365*10)  # 10 years in the future
        
        end_date = st.date_input(
            "End Date", 
            value=default_end,
            min_value=datetime(1985, 1, 1).date(),  # Remove dependency on start_date
            max_value=future_date,  # Allow selection of future dates
            help="Select the end date for ET data (can include future dates for forecasting)",
            key="end_date_input",
            on_change=update_end_date
        )
        
        # Add validation for end date being before start date
        if end_date < start_date:
            st.error("End date cannot be before start date")
            end_date = start_date  # Force end date to be at least start date
            
        # Show warning message if future dates are selected
        today = datetime.now().date()
        if end_date > today:
            st.warning("FUTURE DATES NOT SUPPORTED: The OpenET API does not support future dates and API calls with future dates will not be run. In a future version, we will incorporate ET forecasting, but for now this functionality is a stub.")
    
    with col3:
        # Data interval
        interval = st.selectbox(
            "Data Interval", 
            options=["daily", "monthly"],
            index=0,
            help="Daily data provides more detail but monthly data may load faster for long time periods"
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # ET model selection
        model = st.selectbox(
            "ET Model", 
            options=["Ensemble", "SSEBop", "SIMS", "PTJPL", "eeMETRIC", "DisALEXI"],
            index=0,
            help="Ensemble is the average of all models and recommended for most uses"
        )
    
    with col2:
        # Variable selection (ET or others)
        variable = st.selectbox(
            "Variable", 
            options=["ET", "ETo", "ETof", "NDVI", "PR"],
            index=0,
            help="ET = actual evapotranspiration, ETo = reference ET, PR = precipitation"
        )
    
    with col3:
        # Units selection
        units = st.selectbox(
            "Units", 
            options=["mm", "in"],
            index=0,
            help="Select units for data retrieval"
        )
    
    # API key input
    with st.expander("API Key Settings"):
        api_key = st.text_input(
            "OpenET API Key", 
            value=st.session_state['api_key'],
            type="password",
            help="Enter your OpenET API key. You can get a free key from https://etdata.org"
        )
        st.session_state['api_key'] = api_key
        
        if not api_key:
            st.warning("API key is required to fetch data from OpenET. If you don't have one, you can sign up for free at https://etdata.org")
    
    # Create a tuple of parameters to check if we need to refetch
    current_params = (
        str(st.session_state['user_polygon']), 
        start_date.isoformat(),
        end_date.isoformat(),
        interval,
        model,
        variable,
        units
    )
    
    # Fetch data button
    # Disable the button if no polygon, no API key, or future dates selected
    today = datetime.now().date()
    has_future_dates = end_date > today
    
    fetch_disabled = not st.session_state['user_polygon'] or not api_key
    
    if st.button("Fetch ET Data", disabled=fetch_disabled, type="primary"):
        # Check again for future dates - if selected, show a clear message
        if has_future_dates:
            st.error("API CALL BLOCKED: Cannot fetch data for future dates. Please select an end date no later than today.")
        elif st.session_state['last_query_params'] == current_params and st.session_state['et_data'] is not None:
            st.success("Using cached data. Same parameters as previous query.")
        else:
            with st.spinner("Fetching data from OpenET API..."):
                try:
                    # FUTURE ENHANCEMENT: This is where ET forecasting would be implemented
                    # For now, we only fetch historical data up to the present
                    
                    # Fetch data
                    df = get_et_data_cached(
                        geometry=st.session_state['user_polygon'],
                        start_date=start_date.isoformat(),
                        end_date=min(end_date, today).isoformat(),  # Ensure end date is not in the future
                        interval=interval,
                        model=model,
                        variable=variable,
                        units=units
                    )
                    
                    if df is None or len(df) == 0:
                        st.error("No data returned. This may be due to the area being outside OpenET coverage or no data for the selected period.")
                    else:
                        st.session_state['et_data'] = df
                        st.session_state['last_query_params'] = current_params
                        st.success(f"Data retrieved successfully: {len(df)} data points from {start_date} to {min(end_date, today)}")
                
                except OpenETError as e:
                    st.error(f"Error fetching data: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
    
    return start_date, end_date, interval, model, variable, units

def render_results(start_date, end_date, interval, model, variable, units):
    """Render the results section with visualizations and recommendations."""
    if st.session_state['et_data'] is None:
        return
    
    st.subheader("3. Results and Irrigation Recommendations")
    
    # Get the data
    df = st.session_state['et_data']
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["ET Visualization", "Irrigation Recommendation", "Data Table"])
    
    with tab1:
        st.write(f"**{variable} Data Visualization** ({interval} values from {start_date} to {end_date})")
        
        # Plot time series
        fig = px.line(
            df, 
            x='date', 
            y=variable, 
            title=f"{model} {variable} ({units})",
            labels={'date': 'Date', variable: f"{variable} ({units})"}
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=f"{variable} ({units})",
            hovermode="x unified"
        )
        
        # Add monthly aggregation if data is daily
        if interval == "daily" and len(df) > 31:
            # Create monthly aggregation
            df_monthly = df.copy()
            df_monthly['month'] = df_monthly['date'].dt.to_period('M')
            monthly_values = df_monthly.groupby('month')[variable].sum().reset_index()
            monthly_values['month'] = monthly_values['month'].dt.to_timestamp()
            
            # Add bar chart for monthly totals
            fig2 = px.bar(
                monthly_values, 
                x='month', 
                y=variable,
                title=f"Monthly Total {variable} ({units})",
                labels={'month': 'Month', variable: f"Total {variable} ({units})"}
            )
            
            fig2.update_layout(
                xaxis_title="Month",
                yaxis_title=f"Total {variable} ({units})",
                hovermode="x unified"
            )
            
            # Display both charts
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            # Just display the time series
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.write("**Summary Statistics:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total", f"{df[variable].sum():.1f} {units}")
        with col2:
            st.metric("Average", f"{df[variable].mean():.2f} {units}/{interval}")
        with col3:
            st.metric("Maximum", f"{df[variable].max():.2f} {units}")
        with col4:
            if interval == "daily":
                st.metric("Days", f"{len(df)}")
            else:
                st.metric("Months", f"{len(df)}")
    
    with tab2:
        st.write("**Irrigation Recommendation**")
        
        # Parameters for irrigation recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            mode = st.selectbox(
                "Irrigation Mode", 
                options=["daily", "threshold"],
                index=0,
                help="Daily: replace ET-Rain each day; Threshold: irrigate when deficit reaches threshold"
            )
        
        with col2:
            if mode == "threshold":
                threshold = st.number_input(
                    "Depletion Threshold (mm)", 
                    min_value=10, 
                    max_value=100, 
                    value=30,
                    help="Amount of water depletion (mm) to trigger irrigation"
                )
            else:
                threshold = 25  # default, not used for daily mode
        
        col1, col2 = st.columns(2)
        
        with col1:
            efficiency = st.slider(
                "Irrigation System Efficiency (%)", 
                min_value=50, 
                max_value=100, 
                value=85,
                help="Efficiency of your irrigation system (drip ~ 90%, sprinkler ~ 75%, flood ~ 60%)"
            ) / 100.0
        
        with col2:
            display_units = st.selectbox(
                "Display Units", 
                options=["mm", "inches"],
                index=0 if units == "mm" else 1,
                help="Units for displaying results"
            )
        
        # Check if we have precipitation data
        rain_col = "PR" if "PR" in df.columns else None
        
        # Get irrigation recommendation
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
            
            # Display recommendation summary
            st.info(recommendation["recommendation"])
            
            # Display summary in a nice box
            st.success(summarize_irrigation_needs(recommendation))
            
            # For threshold mode, show scheduled irrigations
            if mode == "threshold" and recommendation["status"] == "success":
                if recommendation.get("schedule") and len(recommendation["schedule"]) > 0:
                    # Create a table of scheduled irrigations
                    schedule_df = pd.DataFrame(recommendation["schedule"])
                    
                    fig = go.Figure(data=[
                        go.Table(
                            header=dict(
                                values=["Date", f"Amount ({display_units})"],
                                fill_color='paleturquoise',
                                align='left'
                            ),
                            cells=dict(
                                values=[
                                    schedule_df['date'], 
                                    schedule_df['amount']
                                ],
                                fill_color='lavender',
                                align='left'
                            )
                        )
                    ])
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No irrigation events scheduled for this period.")
            
            # Visualization of cumulative ET vs irrigation
            if interval == "daily":
                # Create cumulative plots
                df_cum = df.copy()
                df_cum['Cumulative_ET'] = df_cum[variable].cumsum()
                
                # Calculate net ET (ET - rain)
                if rain_col:
                    df_cum['Net_ET'] = (df_cum[variable] - df_cum[rain_col]).clip(lower=0)
                    df_cum['Cumulative_Net_ET'] = df_cum['Net_ET'].cumsum()
                    df_cum['Cumulative_Rain'] = df_cum[rain_col].cumsum()
                
                # Convert to display units if needed
                conversion = 1.0 if display_units == units else (1/25.4 if display_units == "inches" else 25.4)
                for col in df_cum.columns:
                    if col not in ['date'] and df_cum[col].dtype.kind in 'fc':  # float or complex columns
                        df_cum[col] = df_cum[col] * conversion
                
                # Create plot
                fig = go.Figure()
                
                # Add cumulative ET
                fig.add_trace(go.Scatter(
                    x=df_cum['date'], 
                    y=df_cum['Cumulative_ET'],
                    mode='lines',
                    name=f'Cumulative {variable}',
                    line=dict(color='red', width=2)
                ))
                
                # Add rain if available
                if rain_col:
                    fig.add_trace(go.Scatter(
                        x=df_cum['date'], 
                        y=df_cum['Cumulative_Rain'],
                        mode='lines',
                        name='Cumulative Rainfall',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df_cum['date'], 
                        y=df_cum['Cumulative_Net_ET'],
                        mode='lines',
                        name='Cumulative Net ET (ET-Rain)',
                        line=dict(color='orange', width=2)
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"Cumulative Water Balance ({display_units})",
                    xaxis_title="Date",
                    yaxis_title=f"Cumulative Amount ({display_units})",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generating irrigation recommendation: {str(e)}")
    
    with tab3:
        st.write("**Raw Data Table**")
        
        # Convert to display units if different from API units
        display_df = df.copy()
        if display_units != units:
            conversion = 1/25.4 if display_units == "inches" else 25.4
            for col in df.columns:
                if col != 'date' and df[col].dtype.kind in 'fc':  # float or complex columns
                    display_df[col] = df[col] * conversion
        
        # Format date column
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        
        # Show the table
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
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
    1. Draw your field on the map or upload a boundary file
    2. Set the date range and data parameters
    3. Click "Fetch ET Data" to retrieve data
    4. View the results and irrigation recommendations
    """)
    
    st.sidebar.subheader("About")
    st.sidebar.markdown("""
    OpenET provides satellite-based evapotranspiration (ET) data 
    that can be used to estimate crop water use and irrigation needs.
    
    This app demonstrates how this data can be used to create 
    simple irrigation schedules based on ET replacement.
    """)
    
    # Disclaimer
    st.sidebar.subheader("Disclaimer")
    st.sidebar.info("""
    The irrigation recommendations provided by this app are estimates 
    based on ET data and should be used as a guide only. Actual irrigation 
    needs may vary based on specific field conditions, irrigation system 
    efficiency, and local weather patterns.
    """)
    
    # Add link to OpenET
    st.sidebar.markdown("[Learn more about OpenET](https://etdata.org)")
    
    # App version
    st.sidebar.caption("App Version: 1.0.0")

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state
    initialize_session_state()
    
    # App title
    st.title(APP_TITLE)
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    render_map_section()
    
    if st.session_state['user_polygon'] is not None:
        # Render data parameters section
        start_date, end_date, interval, model, variable, units = render_data_parameters()
        
        # Render results if data is available
        if st.session_state['et_data'] is not None:
            render_results(start_date, end_date, interval, model, variable, units)
    
    # Add an expander for cache info (for developers/testing)
    with st.expander("Cache Information", expanded=False):
        st.write("This section shows information about the local cache for developers.")
        
        cache_info = get_cache_info()
        st.write(f"Cache entries: {cache_info['count']}")
        st.write(f"Total cache size: {cache_info['total_size_kb']:.2f} KB")
        
        if st.button("Clear Cache"):
            if clear_cache():
                st.success("Cache cleared successfully")
            else:
                st.error("Failed to clear cache")
            
            # Also clear Streamlit cache
            st.cache_data.clear()
    
    # Footer
    st.write("---")
    st.caption("Powered by OpenET API | Data source: [OpenET](https://etdata.org)")

if __name__ == "__main__":
    main()