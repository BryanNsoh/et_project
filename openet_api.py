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
    api_key: Optional[str] = None
) -> Dict:
    """
    Call the OpenET API to retrieve ET data for a specified geometry and time period.
    
    Args:
        geometry: Coordinates in WGS84 format:
            - For point: [lon, lat]
            - For polygon: [[lon1, lat1], [lon2, lat2], ...] (will be automatically flattened to [lon1, lat1, lon2, lat2, ...])
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Time interval, either "daily" or "monthly"
        model: ET model (Ensemble, SSEBop, SIMS, etc.)
        variable: Variable to retrieve (ET, ETo, ETof, NDVI, PR)
        reference_et: Reference ET source (gridMET, CIMIS)
        units: Units for the result (mm or in)
        api_key: OpenET API key (optional, will load from env if not provided)
    
    Returns:
        Dict: JSON response from the API
        
    Raises:
        OpenETError: For API errors or failed requests
        
    Note:
        For polygon requests, the geometry will be automatically flattened to a 1-dimensional array
        and a "reducer" parameter (default="mean") will be added to the request as required by the API.
    """
    if not api_key:
        api_key = load_api_key()
    
    # Determine if point or polygon
    is_point = isinstance(geometry[0], (int, float))
    
    # Construct URL according to official OpenET documentation
    base_url = "https://openet-api.org"
    if is_point:
        endpoint = f"{base_url}/raster/timeseries/point"
    else:
        endpoint = f"{base_url}/raster/timeseries/polygon"
        
        # For polygon requests, the geometry must be flattened from [[lon1, lat1], [lon2, lat2], ...] 
        # to [lon1, lat1, lon2, lat2, ...] as required by the OpenET API
        if isinstance(geometry[0], list):
            # Flatten the geometry list
            flat_geometry = []
            for point in geometry:
                flat_geometry.extend(point)
            geometry = flat_geometry
            logger.info(f"Flattened polygon geometry for API request")
    
    # Prepare request data according to official OpenET API documentation
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
        payload["reducer"] = "mean"  # Default to mean for polygon queries
        logger.info(f"Added required 'reducer' parameter for polygon query")
    
    # Headers format according to official OpenET documentation
    headers = {
        "Authorization": api_key,  # No "Bearer" prefix
        "Content-Type": "application/json"
    }
    
    # Call API with retry logic
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
                timeout=60  # 60 second timeout
            )
            
            # Log response status for debugging
            logger.info(f"API response status: {response.status_code}")
            
            # Check response status and provide more detailed error logs
            if response.status_code != 200:
                error_message = response.text[:500] if response.text else "No error details provided"
                logger.error(f"API error: Status {response.status_code}, Response: {error_message}")
                
                # Handle specific error codes
                if response.status_code == 404:
                    # This might be an endpoint configuration issue
                    raise OpenETError(f"API endpoint not found (404): {endpoint}. Please verify API documentation for the correct endpoint.")
                
                if response.status_code == 401:
                    raise OpenETError("Authentication failed: API key invalid or expired")
                
                # Check for geometry format errors
                if response.status_code == 422 and "geometry" in error_message:
                    raise OpenETError(f"Invalid geometry format: {error_message}. For polygons, ensure geometry is a flat list of coordinates and 'reducer' parameter is included.")
                
                # Let response.raise_for_status() handle other errors
            
            response.raise_for_status()  # Raise exception for non-200 responses
            
            # Try to parse JSON response
            try:
                data = response.json()
                logger.info(f"API call successful, received response data")
                return data
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response: {response.text[:200]}...")
                raise OpenETError(f"Invalid JSON response from API")
                
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            
            # Check for specific error codes
            if hasattr(e, 'response'):
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
                        time.sleep(5 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        raise OpenETError("API rate limit exceeded. Please try again later.")
            
            # Last retry failed or other error
            if attempt == max_retries - 1:
                raise OpenETError(f"Failed to retrieve data from OpenET API after {max_retries} attempts: {error_msg}")
            
    # Should not reach here, but just in case
    raise OpenETError("Failed to retrieve data from OpenET API")

def parse_openet_response(response_data: Dict, variable: str = "ET") -> pd.DataFrame:
    """
    Parse the OpenET API response into a pandas DataFrame.
    
    Args:
        response_data: JSON response from OpenET API
        variable: The variable name to extract (ET, PR, etc.)
    
    Returns:
        pandas.DataFrame: DataFrame with date and variable columns
        
    Raises:
        OpenETError: If response cannot be parsed
    """
    try:
        # According to OpenET documentation, responses can be in different formats
        # Check for standard API response format first
        timeseries = None
        
        # Check if response has a FeatureCollection structure (typical for polygon queries)
        if response_data.get("type") == "FeatureCollection" and "features" in response_data:
            # Extract timeseries from first feature's properties
            if len(response_data["features"]) > 0:
                properties = response_data["features"][0].get("properties", {})
                timeseries = properties.get("data", properties.get("timeseries", []))
            else:
                raise OpenETError("No features found in API response")
        
        # Check for direct timeseries format (typical for point queries)
        elif "data" in response_data:
            timeseries = response_data.get("data", [])
        
        # Check for legacy format
        elif "timeseries" in response_data:
            timeseries = response_data.get("timeseries", [])
        
        # Check if response itself is the timeseries list
        elif isinstance(response_data, list):
            timeseries = response_data
        
        # If no timeseries found
        if not timeseries:
            logger.error(f"Unexpected API response format: {str(response_data)[:200]}...")
            raise OpenETError("No timeseries data found in API response")
        
        # Convert to DataFrame
        df = pd.DataFrame(timeseries)
        
        # Handle possible column name variations
        date_columns = ["date", "time", "timestamp"]
        date_col = next((col for col in date_columns if col in df.columns), None)
        
        if not date_col:
            raise OpenETError("Date column missing from API response")
            
        # Rename date column to standardize
        if date_col != "date":
            df = df.rename(columns={date_col: "date"})
            
        # Check for variable
        if variable not in df.columns:
            var_col = next((col for col in df.columns if col.upper() == variable.upper()), None)
            if var_col:
                df = df.rename(columns={var_col: variable})
            else:
                logger.warning(f"Variable {variable} not found in response. Available columns: {df.columns.tolist()}")
                raise OpenETError(f"Variable {variable} not found in API response")
        
        # Convert date strings to datetime objects
        df["date"] = pd.to_datetime(df["date"])
        
        # Sort by date
        df = df.sort_values("date")
        
        # Handle missing or negative values
        if df[variable].isna().any():
            logger.warning(f"Found {df[variable].isna().sum()} missing values in {variable} data")
            # Fill missing values with 0 for calculations
            df[variable] = df[variable].fillna(0)
        
        # Convert negative values to 0 (ET shouldn't be negative)
        if (df[variable] < 0).any():
            logger.warning(f"Found {(df[variable] < 0).sum()} negative values in {variable} data")
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
            - For polygon: [[lon1, lat1], [lon2, lat2], ...] (will be automatically flattened)
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
        
    Raises:
        OpenETError: If data cannot be retrieved or parsed
        
    Note:
        For polygon requests, the geometry will be automatically flattened and a 
        reducer parameter will be added to the request as required by the OpenET API.
    """
    # Convert dates to strings if they're datetime objects
    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y-%m-%d")
    
    # ================================================================
    # IMPORTANT: FUTURE DATES NOT SUPPORTED BY OPENET API
    # ================================================================
    # Check if the end date is in the future
    today = datetime.now().date()
    today_str = today.isoformat()
    
    # Convert string dates to date objects for comparison if needed
    end_date_obj = end_date
    if isinstance(end_date, str):
        end_date_obj = datetime.fromisoformat(end_date).date()
    
    # Enforce that we never call the API with future dates
    is_future_query = end_date_obj > today
    
    if is_future_query:
        logger.warning(f"FUTURE DATE DETECTED: {end_date}. OpenET API does not support future dates.")
        # Force end date to today
        end_date = today_str
        logger.info(f"Automatically adjusting end date to today: {today_str}")
    
    try:
        # STUB FOR FUTURE ENHANCEMENT: 
        # In a future version, this is where we would implement ET forecasting
        # For now, we only fetch historical data up to the present
        
        # Call the API with adjusted end date (never in the future)
        response_data = call_openet_api(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,  # This is guaranteed to not be in the future
            interval=interval,
            model=model,
            variable=variable,
            units=units,
            api_key=api_key,
            reducer=reducer
        )
        
        # Parse the response
        df = parse_openet_response(response_data, variable)
        
        # Add a note about the original request if it was a future query
        if is_future_query and not df.empty:
            logger.info(f"Original request included a future end date: {end_date_obj.isoformat()}")
            logger.info(f"Data was fetched up to: {df['date'].max().date().isoformat()}")
        
        return df
    
    except OpenETError as e:
        # No need to check for future date errors since we already handle that
        # by adjusting the end_date before making the API call
        
        # Just log and re-raise the exception
        logger.error(f"Error fetching data from OpenET: {str(e)}")
        raise
    
    except Exception as e:
        # Wrap any other exceptions
        logger.exception("Unexpected error in fetch_openet_data")
        raise OpenETError(f"Failed to fetch OpenET data: {str(e)}")