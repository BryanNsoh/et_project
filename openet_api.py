"""
OpenET API client for fetching evapotranspiration data.
"""
import os
import json
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenETError(Exception):
    """Custom exception for OpenET API related errors."""
    pass

def load_api_key() -> str:
    """
    Load OpenET API key from environment variables or .env file.
    Returns:
        str: API key
    Raises:
        OpenETError: If API key is not found
    """
    # Try to get from environment
    api_key = os.environ.get('OPENET_API_KEY')
    
    # If not in environment, try to load from .env file
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get('OPENET_API_KEY')
        except ImportError:
            logger.warning("python-dotenv not installed, couldn't load from .env file")
    
    if not api_key:
        raise OpenETError("OpenET API key not found. Please set OPENET_API_KEY environment variable or create a .env file.")
    
    return api_key

def call_openet_api(
    geometry: List,
    start_date: str,
    end_date: str,
    interval: str = "daily",
    model: str = "Ensemble",
    variable: str = "ET",
    reference_et: str = "gridMET",
    units: str = "mm",
    api_key: Optional[str] = None,
    reducer: Optional[str] = None
) -> Dict:
    """
    Call the OpenET API to retrieve ET data for a specified geometry and time period.
    
    Args:
        geometry: Coordinates in WGS84 format:
            - For point: [lon, lat]
            - For polygon: [[lon1, lat1], [lon2, lat2], ...] 
              OR [[[lon1, lat1], [lon2, lat2], ...]] (GeoJSON ring).
              These will be flattened into a single list of floats.
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Time interval, either "daily" or "monthly"
        model: ET model (Ensemble, SSEBop, SIMS, etc.)
        variable: Variable to retrieve (ET, ETo, ETof, NDVI, PR)
        reference_et: Reference ET source (gridMET, CIMIS)
        units: Units for the result (mm or in)
        api_key: OpenET API key (optional, will load from env if not provided)
        reducer: Aggregation method for polygon queries (e.g. "mean", "sum").

    Returns:
        Dict: JSON response from the API
        
    Raises:
        OpenETError: For API errors or failed requests
        
    Note:
        For polygon requests, the geometry will be automatically flattened to a 
        1-dimensional array, and a "reducer" parameter is added to the request 
        (default="mean" if not specified).
    """
    if not api_key:
        api_key = load_api_key()
    
    # Determine if this is a point or polygon
    is_point = isinstance(geometry[0], (int, float))
    
    # Construct URL according to official OpenET documentation
    base_url = "https://openet-api.org"
    if is_point:
        endpoint = f"{base_url}/raster/timeseries/point"
    else:
        endpoint = f"{base_url}/raster/timeseries/polygon"
        
        # Handle the possibility of a GeoJSON-like ring (extra nesting)
        # e.g. geometry = [ [ [lon, lat], [lon, lat], ... ] ]
        # We'll unwrap one level if needed:
        if len(geometry) == 1 and isinstance(geometry[0], list) and isinstance(geometry[0][0], list):
            geometry = geometry[0]
        
        # Flatten the polygon coordinates from [[lon1, lat1], [lon2, lat2], ...]
        # to [lon1, lat1, lon2, lat2, ...] as required by the OpenET API
        if isinstance(geometry[0], list):
            flat_geometry = []
            for point in geometry:
                flat_geometry.extend(point)
            geometry = flat_geometry
            logger.info(f"Flattened polygon geometry for API request")
    
    # Prepare request data
    payload = {
        "date_range": [start_date, end_date],
        "interval": interval,
        "geometry": geometry,
        "model": model,
        "variable": variable,
        "reference_et": reference_et,
        "units": units,
        "file_format": "JSON"  # Correct case per documentation
    }
    
    # Add reducer parameter for polygon queries (required by the API)
    if not is_point:
        if not reducer:
            reducer = "mean"  # default
        payload["reducer"] = reducer
        logger.info(f"Added required 'reducer' parameter for polygon query: {reducer}")
    
    # Headers
    headers = {
        "Authorization": api_key,  # No "Bearer" prefix needed
        "Content-Type": "application/json"
    }
    
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            endpoint_type = "point" if is_point else "polygon"
            logger.info(f"Calling OpenET API ({attempt+1}/{max_retries}) - Endpoint: {endpoint} (Type: {endpoint_type})")
            logger.debug(f"Request payload: {json.dumps(payload)}")
            
            response = requests.post(
                endpoint, 
                json=payload, 
                headers=headers,
                timeout=60  # 60-second timeout
            )
            
            # Log response status
            logger.info(f"API response status: {response.status_code}")
            
            # Check for errors
            if response.status_code != 200:
                error_message = response.text[:500] if response.text else "No error details provided"
                logger.error(f"API error: Status {response.status_code}, Response: {error_message}")
                
                if response.status_code == 404:
                    raise OpenETError(f"API endpoint not found (404): {endpoint}. Check documentation.")
                
                if response.status_code == 401:
                    raise OpenETError("Authentication failed: API key invalid or expired")
                
                if response.status_code == 422 and "geometry" in error_message:
                    raise OpenETError(f"Invalid geometry format: {error_message}")
                
            response.raise_for_status()
            
            # Parse JSON
            try:
                data = response.json()
                logger.info(f"API call successful, received response data")
                return data
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response: {response.text[:200]}...")
                raise OpenETError("Invalid JSON response from API")
                
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                if status_code == 401:
                    raise OpenETError("Invalid API key or unauthorized access")
                elif status_code == 422:
                    error_details = e.response.text
                    raise OpenETError(f"Invalid request parameters: {error_details}")
                elif status_code == 429:
                    if attempt < max_retries - 1:
                        import time
                        logger.warning("Rate limit exceeded. Waiting before retry...")
                        time.sleep(5 * (attempt + 1))  # Exponential-ish backoff
                        continue
                    else:
                        raise OpenETError("API rate limit exceeded. Try again later.")
            
            if attempt == max_retries - 1:
                raise OpenETError(f"Failed to retrieve data from OpenET API after {max_retries} attempts: {error_msg}")
    
    # Should not reach here
    raise OpenETError("Failed to retrieve data from OpenET API")

def parse_openet_response(response_data: Union[List, Dict], variable: str = "ET") -> pd.DataFrame:
    """
    Parse the OpenET API response into a pandas DataFrame.
    
    Args:
        response_data: JSON response from OpenET API
                       (can be a dict OR a top-level list)
        variable: The variable name to extract (ET, PR, etc.)
    
    Returns:
        pandas.DataFrame with 'date' and your chosen variable column.
        
    Raises:
        OpenETError if the response doesn't have timeseries data as expected.
    """
    try:
        timeseries = None
        
        # 1) If the entire response is a list, handle that first:
        if isinstance(response_data, list):
            # The API returned a pure list of records (common in point daily queries)
            timeseries = response_data
        
        # 2) Otherwise assume it's a dict; then do your normal checks:
        elif isinstance(response_data, dict):
            # FeatureCollection check (polygon queries)
            if response_data.get("type") == "FeatureCollection" and "features" in response_data:
                if len(response_data["features"]) > 0:
                    props = response_data["features"][0].get("properties", {})
                    timeseries = props.get("data", props.get("timeseries", []))
                else:
                    raise OpenETError("No features found in API response")

            # Direct "data" key check (common in single-point queries)
            elif "data" in response_data:
                timeseries = response_data["data"]

            # "timeseries" key check (some older formats)
            elif "timeseries" in response_data:
                timeseries = response_data["timeseries"]

            # None of the above matched
            else:
                logger.error(f"Unexpected dict format: {str(response_data)[:300]}...")
                raise OpenETError("No recognizable timeseries found in API response")
        else:
            # If it's neither a list nor a dict, we don't know how to parse
            logger.error(f"Invalid response type: {type(response_data).__name__}")
            raise OpenETError("API response is not JSON list or dict")

        # If we still have no timeseries
        if not timeseries:
            logger.error(f"No timeseries data found in API response: {str(response_data)[:300]}...")
            raise OpenETError("No timeseries data found in API response")

        # Create DataFrame
        df = pd.DataFrame(timeseries)

        # Identify date column
        date_columns = ["date", "time", "timestamp"]
        date_col = next((col for col in date_columns if col in df.columns), None)
        if not date_col:
            raise OpenETError("Date column missing from API response")

        # Normalize column name to 'date'
        if date_col != "date":
            df.rename(columns={date_col: "date"}, inplace=True)

        # Ensure we have the requested variable column
        if variable not in df.columns:
            var_col = next((col for col in df.columns if col.lower() == variable.lower()), None)
            if var_col:
                df.rename(columns={var_col: variable}, inplace=True)
            else:
                logger.warning(f"Variable '{variable}' not found. Columns: {df.columns.tolist()}")
                raise OpenETError(f"Variable '{variable}' not found in API response")

        # Convert date strings to actual datetime objects
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df.sort_values("date", inplace=True)

        # Fill any missing (NaN) data
        if df[variable].isna().any():
            logger.warning(f"Found {df[variable].isna().sum()} missing values in {variable}, filling with 0")
            df[variable].fillna(0, inplace=True)

        # Clip negative values
        if (df[variable] < 0).any():
            logger.warning(f"Found {(df[variable] < 0).sum()} negative {variable} values, clipping to 0")
            df[variable] = df[variable].clip(lower=0)

        return df

    except Exception as e:
        if not isinstance(e, OpenETError):
            logger.exception("Error parsing OpenET response")
            raise OpenETError(f"Failed to parse OpenET response: {str(e)}")
        else:
            raise

def fetch_openet_data(
    geometry: List,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    interval: str = "daily",
    model: str = "Ensemble",
    variable: str = "ET",
    units: str = "mm",
    api_key: Optional[str] = None,
    reducer: str = "mean"
) -> pd.DataFrame:
    """
    Fetch ET data from OpenET API and return as a DataFrame.
    This is the main function to call from other modules.
    
    Args:
        geometry: Coordinates in WGS84 format:
            - For point: [lon, lat]
            - For polygon: [[lon, lat], [lon, lat], ...]
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime)
        interval: Time interval ("daily" or "monthly")
        model: ET model to use
        variable: Variable to retrieve
        units: Units for the result
        api_key: OpenET API key
        reducer: Method to reduce/aggregate values for polygon (mean, sum, etc.)
        
    Returns:
        pandas.DataFrame: DataFrame with date and variable columns
    """
    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y-%m-%d")
    
    # Prevent future-date queries
    today = datetime.now().date()
    today_str = today.isoformat()
    end_date_obj = datetime.fromisoformat(end_date).date() if isinstance(end_date, str) else end_date
    
    if end_date_obj > today:
        logger.warning(f"FUTURE DATE DETECTED: {end_date}. Adjusting to {today_str}.")
        end_date = today_str
    
    try:
        # Call the API
        response_data = call_openet_api(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            model=model,
            variable=variable,
            units=units,
            api_key=api_key,
            reducer=reducer
        )
        
        # Parse into DataFrame
        df = parse_openet_response(response_data, variable)
        
        return df
    
    except OpenETError as e:
        logger.error(f"Error fetching data from OpenET: {str(e)}")
        raise
    
    except Exception as e:
        logger.exception("Unexpected error in fetch_openet_data")
        raise OpenETError(f"Failed to fetch OpenET data: {str(e)}")
