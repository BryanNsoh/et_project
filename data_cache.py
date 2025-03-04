"""
Data caching functionality for OpenET data.
"""
import os
import json
import hashlib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def geom_to_key(
    coords: List, 
    start_date: str, 
    end_date: str, 
    interval: str, 
    model: str,
    variable: str
) -> str:
    """
    Generate a unique key for caching based on geometry and query parameters.
    
    Args:
        coords: List of coordinates
        start_date: Start date string
        end_date: End date string
        interval: Data interval
        model: ET model
        variable: Data variable
        
    Returns:
        str: MD5 hash to use as cache key
    """
    # Create a stable representation of geometry
    if isinstance(coords[0], (int, float)):
        # Point: [lon, lat]
        geom_text = json.dumps(coords, separators=(",", ":"))
    else:
        # Polygon: [[[lon1, lat1], ...]]
        geom_text = json.dumps(coords, separators=(",", ":"))
    
    # Combine all parameters for unique key
    params = f"{geom_text}_{start_date}_{end_date}_{interval}_{model}_{variable}"
    return hashlib.md5(params.encode('utf-8')).hexdigest()

def save_cache(
    coords: List,
    start_date: str,
    end_date: str,
    data: pd.DataFrame,
    interval: str = "daily",
    model: str = "Ensemble",
    variable: str = "ET",
) -> str:
    """
    Save data to cache file.
    
    Args:
        coords: Coordinates used for query
        start_date: Start date
        end_date: End date
        data: DataFrame to cache
        interval: Data interval
        model: ET model
        variable: Data variable
        
    Returns:
        str: Cache key used
    """
    # Convert DataFrame to dictionary for JSON serialization
    key = geom_to_key(coords, start_date, end_date, interval, model, variable)
    
    # Prepare data for serialization
    cache_data = {
        "metadata": {
            "geometry_type": "point" if isinstance(coords[0], (int, float)) else "polygon",
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "model": model,
            "variable": variable,
            "cached_at": datetime.now().isoformat(),
        },
        "data": data.to_dict(orient="records"),
    }
    
    filepath = os.path.join(CACHE_DIR, f"{key}.json")
    
    try:
        # Write to a temporary file first, then rename for atomicity
        temp_filepath = filepath + ".tmp"
        with open(temp_filepath, "w") as f:
            json.dump(cache_data, f)
        
        # Atomic rename to avoid partial writes
        os.replace(temp_filepath, filepath)
        logger.info(f"Cached data saved to {filepath}")
        return key
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return None

def load_cache(
    coords: List,
    start_date: str,
    end_date: str,
    interval: str = "daily",
    model: str = "Ensemble",
    variable: str = "ET",
    max_age_hours: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Load data from cache if available and not expired.
    
    Args:
        coords: Coordinates used for query
        start_date: Start date
        end_date: End date
        interval: Data interval
        model: ET model
        variable: Data variable
        max_age_hours: Maximum age of cache in hours, None for no limit
        
    Returns:
        Optional[pd.DataFrame]: Cached data if available, else None
    """
    key = geom_to_key(coords, start_date, end_date, interval, model, variable)
    filepath = os.path.join(CACHE_DIR, f"{key}.json")
    
    if not os.path.exists(filepath):
        logger.info(f"No cache found for key {key}")
        return None
    
    try:
        with open(filepath, "r") as f:
            cache_data = json.load(f)
        
        # Check cache age if max_age specified
        if max_age_hours is not None:
            cached_at = datetime.fromisoformat(cache_data["metadata"]["cached_at"])
            age_hours = (datetime.now() - cached_at).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                logger.info(f"Cache expired (age: {age_hours:.1f} hrs, max: {max_age_hours} hrs)")
                return None
        
        # Convert data back to DataFrame
        df = pd.DataFrame(cache_data["data"])
        
        # Ensure date column is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        logger.info(f"Loaded cached data from {filepath}")
        return df
    
    except Exception as e:
        logger.error(f"Failed to load cache: {e}")
        return None

def clear_cache(specific_key: Optional[str] = None) -> bool:
    """
    Clear cache files.
    
    Args:
        specific_key: If provided, only clear this specific key
        
    Returns:
        bool: True if successful
    """
    try:
        if specific_key:
            filepath = os.path.join(CACHE_DIR, f"{specific_key}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleared specific cache file: {filepath}")
            else:
                logger.warning(f"Cache file not found: {filepath}")
        else:
            # Clear all cache files
            for file in os.listdir(CACHE_DIR):
                if file.endswith('.json'):
                    os.remove(os.path.join(CACHE_DIR, file))
            logger.info("Cleared all cache files")
        return True
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False

def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the current cache.
    
    Returns:
        Dict: Cache statistics
    """
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
        total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files)
        
        # Get metadata from each file
        cache_entries = []
        for filename in cache_files:
            try:
                with open(os.path.join(CACHE_DIR, filename), 'r') as f:
                    data = json.load(f)
                    metadata = data.get('metadata', {})
                    file_size = os.path.getsize(os.path.join(CACHE_DIR, filename))
                    
                    cache_entries.append({
                        'key': filename.replace('.json', ''),
                        'cached_at': metadata.get('cached_at', 'unknown'),
                        'start_date': metadata.get('start_date', 'unknown'),
                        'end_date': metadata.get('end_date', 'unknown'),
                        'model': metadata.get('model', 'unknown'),
                        'size_kb': round(file_size / 1024, 2)
                    })
            except Exception as e:
                logger.error(f"Error reading cache file {filename}: {e}")
        
        return {
            'count': len(cache_files),
            'total_size_kb': round(total_size / 1024, 2),
            'entries': cache_entries
        }
    except Exception as e:
        logger.error(f"Failed to get cache info: {e}")
        return {'error': str(e)}