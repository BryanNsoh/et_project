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
        geometry: Coordinates in WGS84 format for a point [lon, lat] or polygon [[[lon1, lat1], ...]]
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
    """
    if not api_key:
        api_key = load_api_key()
    
    # Determine if point or polygon
    is_point = isinstance(geometry[0], (int, float))
    
    # Construct URL
    base_url = "https://openet-api.org/raster/timeseries"
    endpoint = f"{base_url}/point" if is_point else base_url
    
    # Prepare request data
    payload = {
        "date_range": [start_date, end_date],
        "interval": interval,
        "geometry": geometry,
        "model": model,
        "variable": variable,
        "reference_et": reference_et,
        "units": units,
        "file_format": "JSON"
    }
    
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    
    # Call API with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling OpenET API ({attempt+1}/{max_retries})")
            response = requests.post(
                endpoint, 
                json=payload, 
                headers=headers,
                timeout=60  # 60 second timeout
            )
            response.raise_for_status()  # Raise exception for non-200 responses
            
            # Try to parse JSON response
            try:
                data = response.json()
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
        # Check if response has a FeatureCollection structure
        if response_data.get("type") == "FeatureCollection" and "features" in response_data:
            # Extract timeseries from first feature's properties
            if len(response_data["features"]) > 0:
                properties = response_data["features"][0].get("properties", {})
                timeseries = properties.get("timeseries", [])
            else:
                raise OpenETError("No features found in API response")
        else:
            # Try to directly get timeseries if not in FeatureCollection format
            timeseries = response_data.get("timeseries", [])
            
            # If still not found, check if response itself is the timeseries list
            if not timeseries and isinstance(response_data, list):
                timeseries = response_data
        
        # If no timeseries found
        if not timeseries:
            raise OpenETError("No timeseries data found in API response")
        
        # Convert to DataFrame
        df = pd.DataFrame(timeseries)
        
        # Ensure expected columns exist
        if "date" not in df.columns:
            raise OpenETError("Date column missing from API response")
        if variable not in df.columns:
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
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch ET data from OpenET API and return as a DataFrame.
    This is the main function to call from other modules.
    
    Args:
        geometry: Point coordinates [lon, lat] or polygon [[[lon1, lat1], ...]]
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime)
        interval: Time interval ("daily" or "monthly")
        model: ET model to use
        variable: Variable to retrieve
        units: Units for the result
        api_key: OpenET API key
        
    Returns:
        pandas.DataFrame: DataFrame with date and variable columns
        
    Raises:
        OpenETError: If data cannot be retrieved or parsed
    """
    # Convert dates to strings if they're datetime objects
    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y-%m-%d")
    
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
            api_key=api_key
        )
        
        # Parse the response
        df = parse_openet_response(response_data, variable)
        
        return df
    
    except OpenETError as e:
        # Re-raise the custom exception
        raise
    except Exception as e:
        # Wrap any other exceptions
        logger.exception("Unexpected error in fetch_openet_data")
        raise OpenETError(f"Failed to fetch OpenET data: {str(e)}")