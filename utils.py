"""
Utility functions for the OpenET Irrigation Advisory App.
"""
import os
import json
import tempfile
import zipfile
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import shapely.geometry as sg
from shapely.geometry import shape
import geopandas as gpd
import pandas as pd
import pyproj
from pyproj import Transformer

# Set up logging
logger = logging.getLogger(__name__)

def process_uploaded_geometry(
    file_object: Any,
    max_area_acres: float = 50000
) -> Tuple[List, Optional[Dict]]:
    """
    Process an uploaded geometry file (Shapefile, GeoJSON, or KML).
    
    Args:
        file_object: File object from st.file_uploader
        max_area_acres: Maximum allowed area in acres
        
    Returns:
        Tuple[List, Dict]: List of coordinates and a dict of metadata
    """
    try:
        filename = file_object.name.lower()
        metadata = {"source": "upload", "filename": filename}
        
        # Handle different file types
        if filename.endswith('.zip'):
            # Shapefile (zipped)
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(file_object) as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find .shp file in temp directory
                shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
                if not shp_files:
                    raise ValueError("No .shp file found in the uploaded zip archive")
                
                shp_path = os.path.join(temp_dir, shp_files[0])
                gdf = gpd.read_file(shp_path)
                metadata["original_crs"] = str(gdf.crs)
                
        elif filename.endswith('.geojson'):
            # GeoJSON file
            geojson_data = json.load(file_object)
            gdf = gpd.GeoDataFrame.from_features(
                geojson_data["features"] if "features" in geojson_data else [geojson_data]
            )
            if gdf.crs is None:
                gdf.crs = "EPSG:4326"  # Assume WGS84 if not specified
            metadata["original_crs"] = str(gdf.crs)
            
        elif filename.endswith('.kml'):
            # KML file
            gdf = gpd.read_file(file_object, driver="KML")
            metadata["original_crs"] = "EPSG:4326"  # KML uses WGS84
            
        else:
            raise ValueError(f"Unsupported file format: {filename}. Please upload a .zip (shapefile), .geojson, or .kml file.")
        
        # Ensure we have geometries
        if len(gdf) == 0:
            raise ValueError("No geometries found in the uploaded file")
        
        # Convert to WGS84 if not already
        if gdf.crs and gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        # If there are multiple geometries, take the first one
        # (could be modified to handle multiple or merge them)
        if len(gdf) > 1:
            logger.info(f"File contains {len(gdf)} geometries. Using the first one.")
            metadata["geometry_count"] = len(gdf)
        
        geometry = gdf.geometry.iloc[0]
        
        # Handle different geometry types
        if geometry.geom_type == "Point":
            coords = [geometry.x, geometry.y]
            metadata["geometry_type"] = "point"
        elif geometry.geom_type == "MultiPolygon":
            # Take largest polygon
            largest = max(geometry, key=lambda p: p.area)
            coords = list(largest.exterior.coords)
            metadata["geometry_type"] = "polygon"
        elif geometry.geom_type == "Polygon":
            coords = list(geometry.exterior.coords)
            metadata["geometry_type"] = "polygon"
        else:
            raise ValueError(f"Unsupported geometry type: {geometry.geom_type}. Please provide a point or polygon.")
        
        # Check area for polygons
        if metadata["geometry_type"] == "polygon":
            # Calculate area in square meters
            area_sq_m = calculate_area(coords)
            # Convert to acres
            area_acres = area_sq_m * 0.000247105
            
            metadata["area_acres"] = area_acres
            metadata["area_sq_m"] = area_sq_m
            
            if area_acres > max_area_acres:
                raise ValueError(f"Area exceeds maximum allowed size of {max_area_acres} acres. Please select a smaller area.")
        
        return coords, metadata
    
    except Exception as e:
        logger.exception(f"Error processing uploaded geometry file: {str(e)}")
        raise

def validate_geometry(coords: List, geometry_type: str = "polygon") -> bool:
    """
    Validate a geometry for use with the OpenET API.
    
    Args:
        coords: Coordinate list
        geometry_type: Type of geometry ("point" or "polygon")
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If geometry is invalid with explanation
    """
    if geometry_type == "point":
        # Point validation
        if len(coords) != 2:
            raise ValueError("Invalid point: must have exactly 2 coordinates (longitude, latitude)")
        
        lon, lat = coords
        if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
            raise ValueError(f"Invalid coordinates: longitude must be between -180 and 180, latitude between -90 and 90. Got ({lon}, {lat})")
        
        return True
    
    elif geometry_type == "polygon":
        # Polygon validation
        if len(coords) < 4:
            raise ValueError("Invalid polygon: must have at least 3 points plus closing point")
        
        # Check if first and last points are the same (closed polygon)
        if coords[0] != coords[-1]:
            logger.warning("Polygon not closed. Adding closing point.")
            coords.append(coords[0])
        
        # Check coordinate bounds
        for lon, lat in coords:
            if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                raise ValueError(f"Invalid coordinates: longitude must be between -180 and 180, latitude between -90 and 90. Got ({lon}, {lat})")
        
        # Check for self-intersection
        try:
            polygon = sg.Polygon(coords)
            if not polygon.is_valid:
                # Try to fix with buffer(0)
                fixed = polygon.buffer(0)
                if not fixed.is_valid:
                    raise ValueError("Invalid polygon: self-intersecting or malformed, and could not be fixed automatically")
                logger.warning("Fixed self-intersecting polygon using buffer(0)")
        except Exception as e:
            raise ValueError(f"Invalid polygon: {str(e)}")
        
        return True
    
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")

def calculate_area(coords: List) -> float:
    """
    Calculate the area of a polygon in square meters.
    
    Args:
        coords: List of [lon, lat] coordinates
        
    Returns:
        float: Area in square meters
    """
    try:
        # Create a polygon in WGS84
        polygon = sg.Polygon(coords)
        
        # Create a geodataframe
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        
        # Convert to an equal-area projection for accurate area calculation
        gdf_equal_area = gdf.to_crs("+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs")
        
        # Calculate area in square meters
        area_sq_m = gdf_equal_area.geometry.area.iloc[0]
        
        return area_sq_m
    except Exception as e:
        logger.error(f"Error calculating area: {e}")
        return 0.0

def format_coordinates_for_display(coords: List, precision: int = 5) -> str:
    """
    Format coordinates for display in UI.
    
    Args:
        coords: List of coordinates
        precision: Decimal precision for display
        
    Returns:
        str: Formatted string
    """
    if isinstance(coords[0], (int, float)):
        # Point
        lon, lat = coords
        return f"Point: {lon:.{precision}f}, {lat:.{precision}f}"
    else:
        # Polygon
        vertex_count = len(coords)
        if vertex_count > 0 and coords[0] == coords[-1]:
            vertex_count -= 1  # Don't count the closing point twice
        
        first_point = coords[0]
        return f"Polygon with {vertex_count} vertices, starting at {first_point[0]:.{precision}f}, {first_point[1]:.{precision}f}"

def get_random_demo_field() -> List:
    """
    Get a random demonstration field for showcasing the app.
    
    Returns:
        List: Coordinates for a sample field
    """
    # Example field in California's Central Valley (almond orchard)
    return [
        [-119.45104, 36.85125],
        [-119.44805, 36.85123],
        [-119.44803, 36.84906],
        [-119.45102, 36.84908],
        [-119.45104, 36.85125]
    ]