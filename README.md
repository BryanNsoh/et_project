# OpenET Irrigation Advisory App - Developer Guide

## Project Overview

The OpenET Irrigation Advisory App is a Streamlit-based web application that provides data-driven irrigation recommendations using satellite-derived evapotranspiration (ET) data. The app creates a bridge between advanced remote sensing data and practical on-farm irrigation decisions.

**Key features:**
- Interactive map interface for field selection (draw or upload boundaries)
- ET data retrieval from OpenET API with caching
- Daily and threshold-based irrigation scheduling 
- Data visualization with charts and reports
- Water balance tracking for irrigation management

## Quick Start

```bash
# Clone repository (replace with actual URL)
git clone <repository-url>
cd openet-irrigation-advisory

# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with API keys
echo "OPENET_API_KEY=your_key_here" > .env
echo "GEOAPIFY_API_KEY=your_key_here" >> .env

# Run the application
streamlit run app.py
```

The app will be available at http://localhost:8501

## Core Components & Architecture

### Main Components

1. **`app.py`**: Streamlit application controller and UI
2. **`openet_api.py`**: OpenET API client for ET data retrieval
3. **`data_cache.py`**: Caching system for API responses
4. **`irrigation.py`**: Irrigation scheduling algorithms
5. **`utils.py`**: Geometry processing and utility functions

### Data Flow

```
User Selection → Geometry Validation → API Request/Cache Check → ET Data Processing 
→ Irrigation Calculation → Visualization → Recommendations
```

### Critical Functionality

#### Field Selection
The app uses Folium for map rendering and polygon drawing/editing:
- Draw tools are in `create_map()` function in `app.py`
- Coordinates are stored in `st.session_state['user_polygon']`
- Geometry validation in `utils.py:validate_geometry()`

#### OpenET API Integration
The OpenET API client handles requests for ET data:
- Authentication via API key from `.env` file
- Specific handling for point vs. polygon queries
- Robust error handling and retry logic

#### Data Caching
To minimize API calls, responses are cached:
- Cache is keyed on geometry, date range, and query parameters
- Files stored in `cache/` directory as JSON
- Cache expiration configurable (default: 24 hours)

#### Irrigation Calculations
Two primary irrigation scheduling methods:
1. **Daily Replacement**: Irrigate to replace daily ET
2. **Threshold-Based**: Trigger irrigation when accumulated deficit reaches threshold

## Detailed Component Reference

### Streamlit UI (`app.py`)

```python
# Key functions:

def initialize_session_state():
    """Sets up state variables for field selection, dates, etc."""
    
def create_map(center=None, zoom=None):
    """Creates interactive map with drawing tools"""
    
def render_sidebar():
    """Creates sidebar with date pickers and settings"""
    
def fetch_data(params):
    """Retrieves ET data from API or cache"""
    
def render_results(params):
    """Creates charts and recommendations"""
```

**Session State Variables:**
- `user_polygon`: Selected field coordinates
- `geometry_metadata`: Field information (area, type)  
- `et_data`: Retrieved ET data
- `start_date`/`end_date`: Analysis period
- `map_center`/`map_zoom`: Map view state
- `draw_mode`: Whether drawing is enabled

### API Client (`openet_api.py`)

```python
# Main functions:

def fetch_openet_data(geometry, start_date, end_date, **kwargs):
    """Main entry point for retrieving ET data"""

def call_openet_api(geometry, start_date, end_date, **kwargs):
    """Makes actual API requests with retries"""
    
def parse_openet_response(response_data, variable="ET"):
    """Converts API responses to pandas DataFrames"""
```

**API Endpoints:**
- Point data: `https://openet-api.org/raster/timeseries/point`
- Polygon data: `https://openet-api.org/raster/timeseries/polygon`

**Required Parameters:**
- `geometry`: Point [lon, lat] or polygon [[lon1, lat1], [lon2, lat2], ...]
- `date_range`: [start_date, end_date]
- `interval`: "daily" or "monthly"
- `model`: "Ensemble", "SSEBop", "SIMS", etc.
- `variable`: "ET", "ETo", "ETof", "NDVI", "PR"
- `units`: "mm" or "in"

### Irrigation Calculations (`irrigation.py`)

```python
# Main functions:

def get_irrigation_recommendation(df, mode="daily", **kwargs):
    """Generate irrigation recommendations from ET data"""
    
def compute_daily_replacement(df, et_col="ET", rain_col=None, efficiency=1.0):
    """Calculate daily irrigation needs"""
    
def compute_threshold_schedule(df, threshold_mm, **kwargs):
    """Create threshold-based irrigation schedule"""
```

**Recommendation Output:**
- Recommendation text
- Total irrigation amount
- Daily values or scheduled events
- Water balance summary

## Common Development Tasks

### Adding a New ET Model

1. Update the model selection in `app.py`:
```python
model = st.selectbox(
    "ET Model", 
    options=["Ensemble", "SSEBop", "SIMS", "PTJPL", "eeMETRIC", "DisALEXI", "YOUR_NEW_MODEL"],
    index=0
)
```

2. No changes needed to API client - it will pass the model parameter directly

### Modifying the Irrigation Algorithm

Example: Adding a crop coefficient adjustment:

```python
# In irrigation.py
def compute_daily_replacement(df, et_col="ET", rain_col=None, efficiency=1.0, crop_coef=1.0):
    """Compute daily irrigation needs with crop coefficient."""
    result_df = df.copy()
    
    # Apply crop coefficient to ET
    result_df["Adjusted_ET"] = df[et_col] * crop_coef
    
    # Calculate net ET (ET minus rainfall)
    if rain_col and rain_col in df.columns:
        result_df["Net_ET"] = (result_df["Adjusted_ET"] - df[rain_col]).clip(lower=0)
    else:
        result_df["Net_ET"] = result_df["Adjusted_ET"]
    
    # Calculate irrigation need considering efficiency
    if efficiency < 1.0:
        result_df["Irrigation_mm"] = result_df["Net_ET"] / efficiency
    else:
        result_df["Irrigation_mm"] = result_df["Net_ET"]
    
    return result_df
```

Then update the UI in `app.py` to add the crop coefficient input.

### Adding a New Data Source

To incorporate additional data (e.g., weather forecasts):

1. Create a new module (e.g., `weather_api.py`) following the pattern of `openet_api.py`
2. Add API key to `.env` file
3. Add caching for the new data source
4. Update the UI to display the new data
5. Integrate the data into the irrigation calculations

## Troubleshooting Guide

### API Connection Issues

**Problem**: OpenET API returns 401 errors
- **Check**: Verify API key in `.env` file
- **Solution**: Register/renew API key at etdata.org

**Problem**: No data returned for valid field
- **Check**: Ensure field is within OpenET coverage area (Western US)
- **Check**: Verify date range (typically 2016 onward)
- **Check**: API request in the browser dev tools network tab
- **Solution**: Try a smaller field or different time period

### Performance Issues

**Problem**: Slow response for large fields
- **Cause**: Large polygon areas require processing more satellite data
- **Solution**: Limit field size or implement backend processing

**Problem**: App becomes unresponsive with large date ranges
- **Cause**: Too much data being processed in browser
- **Solution**: Implement data aggregation or pagination

### Cache Management

**Problem**: Cache grows too large
- **Solution**: Add a cache cleanup function:

```python
# Add to data_cache.py
def cleanup_old_cache(max_age_days=30, max_size_mb=500):
    """Remove old cache files."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    
    # Remove files older than max_age_days
    for file in os.listdir(cache_dir):
        if not file.endswith('.json'):
            continue
        
        file_path = os.path.join(cache_dir, file)
        file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))).days
        
        if file_age_days > max_age_days:
            os.remove(file_path)
            logger.info(f"Removed old cache file: {file}")
```

## Deployment Considerations

### Environment Setup

For production deployment:
- Use `requirements-prod.txt` (create this with only production dependencies)
- Set up proper logging to file rather than console
- Configure HTTPS for security

### Streamlit Deployment Options

1. **Streamlit Cloud** (streamlit.io):
   - Connect to GitHub repository
   - Add secrets for API keys
   - Limited to 1GB RAM

2. **Docker Container**:
   - Create a Dockerfile:
   ```
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   
   EXPOSE 8501
   
   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]
   ```
   - Build: `docker build -t openet-irrigation .`
   - Run: `docker run -p 8501:8501 --env-file .env openet-irrigation`

3. **Server Deployment**:
   - Use Nginx as a reverse proxy
   - Set up systemd service for auto-restart
   - Use supervisor or PM2 for process management

## Data & Database Considerations

Current implementation uses file-based caching. For large-scale deployment:

1. Consider implementing a database backend:
   - MongoDB for flexible JSON storage
   - PostgreSQL/PostGIS for spatial data

2. Separate caching into Redis for better performance

Example Redis implementation:
```python
# Replace file-based caching with Redis
import redis

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def save_cache(coords, start_date, end_date, data, **kwargs):
    """Save data to Redis cache."""
    key = geom_to_key(coords, start_date, end_date, **kwargs)
    
    # Convert DataFrame to JSON string
    data_json = data.to_json(orient="records", date_format="iso")
    
    # Store in Redis with expiration (24 hours)
    redis_client.setex(key, 86400, data_json)
    
    return key

def load_cache(coords, start_date, end_date, **kwargs):
    """Load data from Redis cache."""
    key = geom_to_key(coords, start_date, end_date, **kwargs)
    
    # Try to get data from Redis
    data_json = redis_client.get(key)
    
    if data_json:
        # Convert JSON back to DataFrame
        return pd.read_json(data_json, orient="records")
    
    return None
```

## Common Customization Examples

### Adding Custom Field Statistics

```python
# In utils.py
def calculate_field_statistics(geometry):
    """Calculate advanced statistics for a field."""
    # Create shapely polygon
    polygon = sg.Polygon(geometry)
    
    # Calculate perimeter in meters
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    gdf_equal_area = gdf.to_crs("+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96")
    
    stats = {
        "area_acres": gdf_equal_area.area.iloc[0] * 0.000247105,
        "perimeter_m": gdf_equal_area.length.iloc[0],
        "centroid": [polygon.centroid.x, polygon.centroid.y],
        "compactness": 4 * math.pi * polygon.area / (polygon.length ** 2)
    }
    
    return stats
```

Then display in the UI:
```python
if st.session_state['user_polygon']:
    stats = calculate_field_statistics(st.session_state['user_polygon'])
    st.sidebar.subheader("Field Statistics")
    st.sidebar.metric("Area", f"{stats['area_acres']:.2f} acres")
    st.sidebar.metric("Perimeter", f"{stats['perimeter_m']:.0f} m")
    st.sidebar.metric("Compactness", f"{stats['compactness']:.2f}")
```

### Adding Export Functionality

```python
# In app.py
def export_data(df, filename="et_data.csv"):
    """Export data to CSV file."""
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="Export Data",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# In the results section:
if st.session_state['et_data'] is not None:
    export_data(st.session_state['et_data'])
```

## Final Recommendations

1. **Documentation**: Keep inline comments updated as you modify the code
2. **Testing**: Run `python -m unittest discover -s tests` after changes
3. **Error Handling**: Surface clear error messages to users
4. **Performance**: Consider background processing for large computations
5. **Security**: Keep API keys secure in environment variables
6. **Maintenance**: Update dependencies regularly and check for OpenET API changes

This concludes the developer guide for the OpenET Irrigation Advisory App. With this information, you should be able to understand, modify, and extend the application to meet your specific needs.
