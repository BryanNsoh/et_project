"""
Unit tests for the OpenET API client.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import json
import pandas as pd
from datetime import datetime

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openet_api import OpenETError, call_openet_api, parse_openet_response, fetch_openet_data

class TestOpenETAPI(unittest.TestCase):
    """Test cases for OpenET API client."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample point coordinates (California almond orchard)
        self.point_coords = [-119.45104, 36.85125]
        
        # Sample polygon coordinates (small field)
        self.polygon_coords = [
            [-119.45104, 36.85125],
            [-119.44805, 36.85123],
            [-119.44803, 36.84906],
            [-119.45102, 36.84908],
            [-119.45104, 36.85125]
        ]
        
        # Sample API response
        self.sample_response = {
            "type": "FeatureCollection",
            "features": [{
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [self.polygon_coords]
                },
                "properties": {
                    "timeseries": [
                        {"date": "2023-01-01", "ET": 1.2},
                        {"date": "2023-01-02", "ET": 1.5},
                        {"date": "2023-01-03", "ET": 1.8}
                    ],
                    "units": "mm",
                    "model": "Ensemble",
                    "variable": "ET"
                }
            }]
        }
        
        # Sample API response for point
        self.sample_point_response = {
            "timeseries": [
                {"date": "2023-01-01", "ET": 1.2},
                {"date": "2023-01-02", "ET": 1.5},
                {"date": "2023-01-03", "ET": 1.8}
            ],
            "units": "mm",
            "model": "Ensemble",
            "variable": "ET"
        }
    
    @patch('openet_api.requests.post')
    def test_call_openet_api_point(self, mock_post):
        """Test calling API with point coordinates."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_point_response
        mock_post.return_value = mock_response
        
        # Call function
        result = call_openet_api(
            geometry=self.point_coords,
            start_date="2023-01-01",
            end_date="2023-01-03",
            interval="daily",
            model="Ensemble",
            variable="ET",
            units="mm",
            api_key="test_api_key"
        )
        
        # Check result
        self.assertEqual(result, self.sample_point_response)
        
        # Check that requests.post was called correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        # Check URL (should use point endpoint)
        self.assertEqual(args[0], "https://openet-api.org/raster/timeseries/point")
        
        # Check headers
        self.assertEqual(kwargs['headers']['Authorization'], "test_api_key")
        self.assertEqual(kwargs['headers']['Content-Type'], "application/json")
        
        # Check payload
        payload = kwargs['json']
        self.assertEqual(payload['geometry'], self.point_coords)
        self.assertEqual(payload['date_range'], ["2023-01-01", "2023-01-03"])
        self.assertEqual(payload['interval'], "daily")
        self.assertEqual(payload['model'], "Ensemble")
        self.assertEqual(payload['variable'], "ET")
        self.assertEqual(payload['units'], "mm")
    
    @patch('openet_api.requests.post')
    def test_call_openet_api_polygon(self, mock_post):
        """Test calling API with polygon coordinates."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_response
        mock_post.return_value = mock_response
        
        # Call function
        result = call_openet_api(
            geometry=self.polygon_coords,
            start_date="2023-01-01",
            end_date="2023-01-03",
            interval="daily",
            model="Ensemble",
            variable="ET",
            units="mm",
            api_key="test_api_key"
        )
        
        # Check result
        self.assertEqual(result, self.sample_response)
        
        # Check that requests.post was called correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        # Check URL (should use base endpoint for polygon)
        self.assertEqual(args[0], "https://openet-api.org/raster/timeseries")
        
        # Check payload
        payload = kwargs['json']
        self.assertEqual(payload['geometry'], self.polygon_coords)
    
    @patch('openet_api.requests.post')
    def test_api_error_handling(self, mock_post):
        """Test handling of API errors."""
        # Mock response for 401 error
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Exception("401 Client Error: Unauthorized")
        mock_post.return_value = mock_response
        
        # Call function and check for error
        with self.assertRaises(OpenETError) as context:
            call_openet_api(
                geometry=self.point_coords,
                start_date="2023-01-01",
                end_date="2023-01-03",
                interval="daily",
                model="Ensemble",
                variable="ET",
                units="mm",
                api_key="invalid_api_key"
            )
        
        self.assertIn("Invalid API key", str(context.exception))
    
    def test_parse_openet_response_feature_collection(self):
        """Test parsing API response in FeatureCollection format."""
        # Parse the sample response
        df = parse_openet_response(self.sample_response, "ET")
        
        # Check dataframe
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ['date', 'ET'])
        self.assertEqual(df['ET'].tolist(), [1.2, 1.5, 1.8])
        
        # Check date conversion
        self.assertIsInstance(df['date'].iloc[0], pd.Timestamp)
    
    def test_parse_openet_response_direct_timeseries(self):
        """Test parsing API response with direct timeseries format."""
        # Parse the sample point response
        df = parse_openet_response(self.sample_point_response, "ET")
        
        # Check dataframe
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ['date', 'ET'])
        self.assertEqual(df['ET'].tolist(), [1.2, 1.5, 1.8])
    
    def test_parse_openet_response_missing_data(self):
        """Test parsing API response with missing data column."""
        # Create response with missing column
        bad_response = {
            "timeseries": [
                {"date": "2023-01-01", "OTHER": 1.2},
                {"date": "2023-01-02", "OTHER": 1.5}
            ]
        }
        
        # Check that it raises an error
        with self.assertRaises(OpenETError) as context:
            parse_openet_response(bad_response, "ET")
        
        self.assertIn("Variable ET not found", str(context.exception))
    
    @patch('openet_api.call_openet_api')
    def test_fetch_openet_data(self, mock_call_api):
        """Test the main fetch_openet_data function."""
        # Mock API call
        mock_call_api.return_value = self.sample_response
        
        # Call function
        df = fetch_openet_data(
            geometry=self.polygon_coords,
            start_date="2023-01-01",
            end_date="2023-01-03",
            interval="daily",
            model="Ensemble",
            variable="ET",
            units="mm",
            api_key="test_api_key"
        )
        
        # Check result
        self.assertEqual(len(df), 3)
        self.assertEqual(df['ET'].tolist(), [1.2, 1.5, 1.8])
        
        # Check API was called with correct parameters
        mock_call_api.assert_called_once_with(
            geometry=self.polygon_coords,
            start_date="2023-01-01",
            end_date="2023-01-03",
            interval="daily",
            model="Ensemble",
            variable="ET",
            units="mm",
            api_key="test_api_key"
        )
    
    @patch('openet_api.call_openet_api')
    def test_fetch_openet_data_with_datetime_objects(self, mock_call_api):
        """Test fetch_openet_data with datetime objects."""
        # Mock API call
        mock_call_api.return_value = self.sample_response
        
        # Call function with datetime objects
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)
        
        df = fetch_openet_data(
            geometry=self.polygon_coords,
            start_date=start_date,
            end_date=end_date,
            interval="daily",
            model="Ensemble",
            variable="ET",
            units="mm",
            api_key="test_api_key"
        )
        
        # Check API was called with correct string dates
        mock_call_api.assert_called_once_with(
            geometry=self.polygon_coords,
            start_date="2023-01-01",
            end_date="2023-01-03",
            interval="daily",
            model="Ensemble",
            variable="ET",
            units="mm",
            api_key="test_api_key"
        )

if __name__ == '__main__':
    unittest.main()